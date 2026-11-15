"""Frozen AST: general and reference-conditioned regressors on identical queries.

Run extract, then fit, then evaluate once. No audio or metadata is uploaded.
"""
import argparse
import collections
import contextlib
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL = 'MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION = 'f826b80d28226b62986cc218e5cec390b1096902'
BASE = Path(__file__).resolve().parent / 'base'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device(requested='auto'):
    if requested == 'auto':
        requested = ('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested Apple GPU is unavailable')
    return torch.device(requested)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def extract(args, manifest):
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import ASTFeatureExtractor, ASTModel
    target = device(args.device)
    if args.cache.exists():
        raise FileExistsError(args.cache)
    files = {}
    for p in args.audio.rglob('*.wav'):
        if p.stem in files:
            raise ValueError(f'Duplicate clip filename: {p.stem}')
        files[p.stem] = p
    required = sorted({r['clip'] for e in manifest['examples']
                       for r in [e['query'], *e['references']]})
    missing = set(required) - files.keys()
    if missing:
        raise FileNotFoundError(f'{len(missing)} required WAV files missing; first: {sorted(missing)[:3]}')
    extractor = ASTFeatureExtractor.from_pretrained(BASE, local_files_only=True)
    encoder = ASTModel.from_pretrained(BASE, local_files_only=True).to(target).eval()
    encoder.requires_grad_(False)
    partial = args.cache.with_suffix('.partial.pt')
    embeddings = {}
    if partial.exists():
        previous = load(partial)
        if previous['manifest_sha256'] != sha(args.manifest) or previous['model'] != MODEL:
            raise ValueError('Partial embedding cache differs from requested run')
        if previous['revision'] != REVISION:
            raise ValueError('Encoder revision changed since partial extraction')
        embeddings = previous['embeddings']
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    def save_cache(path):
        temporary = path.with_suffix('.writing.pt')
        torch.save({'embeddings': embeddings, 'manifest_sha256': sha(args.manifest),
                    'model': MODEL, 'revision': REVISION,
                    'device': str(target)}, temporary)
        temporary.replace(path)

    started, completed = time.monotonic(), 0
    print(f'Extracting on {target}; {len(embeddings)}/{len(required)} already cached', flush=True)
    with torch.inference_mode():
        for i, key in enumerate(required):
            if key in embeddings:
                continue
            waveform, rate = sf.read(files[key], dtype='float32', always_2d=True)
            waveform = waveform.mean(axis=1)
            if rate != 16000:
                gcd = math.gcd(rate, 16000)
                waveform = resample_poly(waveform, 16000 // gcd, rate // gcd)
            if not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError(f'Invalid audio: {key}')
            inputs = extractor(waveform, sampling_rate=16000, return_tensors='pt')
            inputs = {k: v.to(target) for k, v in inputs.items()}
            precision = (torch.autocast('cuda', dtype=torch.float16)
                         if target.type == 'cuda' else contextlib.nullcontext())
            with precision:
                embedding = encoder(**inputs).pooler_output[0]
            if not torch.isfinite(embedding).all():
                raise RuntimeError(f'Non-finite AST embedding: {key}')
            embeddings[key] = embedding.float().cpu()
            completed += 1
            if completed == 1 or completed % 25 == 0:
                save_cache(partial)
                seconds = (time.monotonic() - started) / completed
                print(f'Embedded {len(embeddings)}/{len(required)}; {seconds:.2f} sec/clip', flush=True)
    save_cache(args.cache)
    print(f'Saved {len(embeddings)} frozen embeddings', flush=True)


class Regressor(nn.Module):
    def __init__(self, width, personal):
        super().__init__()
        self.personal = personal
        size = width * 3 + 1 if personal else width
        self.mlp = nn.Sequential(nn.Linear(size, 128), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, query, refs, scores):
        if not self.personal:
            return self.mlp(query).squeeze(-1)
        expanded = query[:, None, :].expand_as(refs)
        features = torch.cat((expanded, refs, expanded - refs, (scores / 10)[..., None]), dim=-1)
        delta = self.mlp(features).squeeze(-1)
        return (scores + delta).mean(dim=1)


def tensors(examples, embeddings):
    return (torch.stack([embeddings[e['query']['clip']] for e in examples]),
            torch.stack([torch.stack([embeddings[r['clip']] for r in e['references']]) for e in examples]),
            torch.tensor([[r['score'] for r in e['references']] for e in examples]),
            torch.tensor([e['query']['score'] for e in examples]))


def participant_weights(examples):
    counts = collections.Counter(e['participant'] for e in examples)
    return torch.tensor([1 / counts[e['participant']] for e in examples])


def metrics(examples, predictions):
    by_person, by_band = collections.defaultdict(list), collections.defaultdict(list)
    rows = []
    for e, pred in zip(examples, predictions):
        target = e['query']['score']
        pred = float(pred)
        error = abs(pred - target)
        by_person[e['participant']].append(error)
        band = 'low' if target <= 3 else 'moderate' if target <= 6 else 'high'
        by_band[band].append(error)
        rows.append({'participant': e['participant'], 'clip': e['query']['clip'],
                     'target': target, 'prediction': pred})
    per_person = {p: float(np.mean(v)) for p, v in by_person.items()}
    return {'participant_macro_mae': float(np.mean(list(per_person.values()))),
            'per_participant_mae': per_person,
            'per_band': {b: {'count': len(v), 'mae': float(np.mean(v))} for b, v in by_band.items()},
            'outside_scale_count': sum(not 1 <= r['prediction'] <= 10 for r in rows),
            'predictions': rows}


def cached(args):
    cache = load(args.cache)
    if cache['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Embedding cache and manifest differ')
    return cache


def normalize(data, mean, std, target):
    q, r, s, y = data
    return ((q - mean).div(std).to(target), (r - mean).div(std).to(target),
            s.to(target), y.to(target))


def fit(args, manifest):
    target = device(args.device)
    cache = cached(args)
    train = [e for e in manifest['examples'] if e['split'] == 'train']
    val = [e for e in manifest['examples'] if e['split'] == 'validation']
    data = tensors(train, cache['embeddings'])
    # Statistics use training participants only, one observation per unique clip.
    keys = sorted({r['clip'] for e in train for r in [e['query'], *e['references']]})
    matrix = torch.stack([cache['embeddings'][k] for k in keys])
    mean, std = matrix.mean(0), matrix.std(0).clamp_min(1e-4)
    q, r, s, y = normalize(data, mean, std, target)
    vq, vr, vs, _ = normalize(tensors(val, cache['embeddings']), mean, std, target)
    weights = participant_weights(train).to(target)
    weights = weights / weights.mean()
    for personal in (False, True):
        name = 'personal' if personal else 'general'
        checkpoint = args.output / f'{name}.pt'
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        seed_all(args.seed)
        model = Regressor(q.shape[1], personal).to(target)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.HuberLoss(delta=1., reduction='none')
        best, stale, history = float('inf'), 0, []
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(train), device=target)
            loss_sum = 0.
            for batch in order.split(64):
                opt.zero_grad(set_to_none=True)
                prediction = model(q[batch], r[batch], s[batch])
                loss = (loss_fn(prediction, y[batch]) * weights[batch]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach()) * len(batch)
            model.eval()
            with torch.inference_mode():
                predictions = model(vq, vr, vs).cpu().tolist()
            metric = metrics(val, predictions)['participant_macro_mae']
            record = {'model': name, 'epoch': epoch + 1,
                      'train_huber': loss_sum / len(train), 'validation_macro_mae': metric}
            history.append(record)
            print(json.dumps(record), flush=True)
            if metric < best:
                best, stale = metric, 0
                torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                            'personal': personal, 'width': q.shape[1], 'mean': mean, 'std': std,
                            'manifest_sha256': sha(args.manifest), 'seed': args.seed,
                            'best_epoch': epoch + 1, 'validation_mae': metric,
                            'encoder_model': cache['model'], 'encoder_revision': cache['revision']}, checkpoint)
            else:
                stale += 1
            if stale >= 10:
                break
        (args.output / f'{name}_history.json').write_text(json.dumps(history, indent=2))
    (args.output / 'environment.json').write_text(json.dumps({
        'torch': str(torch.__version__), 'device': str(target),
        'gpu': torch.cuda.get_device_name() if target.type == 'cuda' else None,
        'packages': {p: importlib.metadata.version(p) for p in
                     ['torch', 'transformers', 'numpy', 'scipy', 'soundfile']},
        'training_code_sha256': sha(__file__), 'seed': args.seed,
        'max_epochs': args.epochs, 'batch_size': 64, 'learning_rate': 1e-3,
        'weight_decay': 1e-3, 'early_stopping_patience': 10, 'huber_delta': 1.,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if target.type == 'cuda' else None}, indent=2))


def evaluate(args, manifest):
    cache = cached(args)
    report_file = args.output / 'test_results.json'
    if report_file.exists():
        raise FileExistsError('Test results already exist; preserve the original evaluation.')
    examples = [e for e in manifest['examples'] if e['split'] == 'test']
    data = tensors(examples, cache['embeddings'])
    results = {}
    for name in ('general', 'personal'):
        checkpoint = load(args.output / f'{name}.pt')
        if checkpoint['manifest_sha256'] != sha(args.manifest):
            raise ValueError('Checkpoint and split differ')
        model = Regressor(checkpoint['width'], checkpoint['personal']).eval()
        model.load_state_dict(checkpoint['state_dict'])
        q, r, s, _ = normalize(data, checkpoint['mean'], checkpoint['std'], 'cpu')
        with torch.inference_mode():
            predictions = model(q, r, s).tolist()
        results[name] = metrics(examples, predictions)
    results['reference_mean'] = metrics(examples, data[2].mean(1).tolist())
    results['last_reference'] = metrics(examples, data[2][:, -1].tolist())
    train = [e for e in manifest['examples'] if e['split'] == 'train']
    train_means = collections.defaultdict(list)
    for e in train:
        train_means[e['participant']].append(e['query']['score'])
    constant = float(np.mean([np.mean(v) for v in train_means.values()]))
    results['training_mean'] = metrics(examples, [constant] * len(examples))
    pids = sorted(results['general']['per_participant_mae'])
    differences = np.array([results['general']['per_participant_mae'][p] -
                            results['personal']['per_participant_mae'][p] for p in pids])
    rng = np.random.default_rng(args.seed)
    bootstrap = rng.choice(differences, size=(10000, len(pids)), replace=True).mean(1)
    results['personalization_improvement'] = {'mean_mae_reduction': float(differences.mean()),
                                            'participant_bootstrap_95_ci': np.quantile(bootstrap, [.025, .975]).tolist(),
                                            'test_participants': len(pids)}
    report_file.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v.get('participant_macro_mae', v) for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['extract', 'fit', 'evaluate'])
    parser.add_argument('--manifest', type=Path, default=Path('runs/ast_reference/manifest.json'))
    parser.add_argument('--audio', type=Path, default=Path('data/tame/mic1_trim_v1'))
    parser.add_argument('--cache', type=Path, default=Path('runs/ast_reference/embeddings.pt'))
    parser.add_argument('--output', type=Path, default=Path('runs/ast_reference'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    seed_all(args.seed)
    {'extract': extract, 'fit': fit, 'evaluate': evaluate}[args.stage](args, manifest)
