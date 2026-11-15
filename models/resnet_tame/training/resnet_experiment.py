"""ImageNet-initialized AVPENet-style ResNet-34: frozen vs final-block training.

Uses the existing TAME participant split and query recordings. This is a new
experiment with ImageNet initialization; checkpoint provenance is saved explicitly.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from train import device, load, metrics, participant_weights, seed_all, sha


SETTINGS = {
    'initialization': 'torchvision.ResNet34_Weights.IMAGENET1K_V1',
    'encoder': 'resnet34_single_channel_global_average_pool_512',
    'variants': ['frozen', 'finetuned'],
    'head': [512, 128, 64, 1], 'dropout': .2,
    'seed': 42, 'max_epochs': 100, 'patience': 10, 'batch_size': 32,
    'head_lr': 1e-3, 'encoder_lr': 1e-5, 'weight_decay': 1e-3,
    'loss': 'participant_weighted_huber', 'huber_delta': 1.,
    'selection': 'participant_macro_validation_mae',
    'batch_norm': 'fixed_running_statistics; layer4 affine trainable in finetuned',
    'sample_rate': 16000, 'n_fft': 400, 'hop_length': 160,
    'n_mels': 128, 'fmin': 50, 'fmax': 8000, 'frames': 300,
    'preprocessing': 'mono; polyphase resample; peak normalization; log mel; '
                     'per-clip minmax; right zero-pad or keep first 300 frames',
    'denoising': False, 'augmentation': False,
    'outputs': 'unconstrained pain score; out-of-range counts reported',
    'reference_use': 'reference scores used only for evaluation controls',
    'architecture_source': 'https://github.com/oussama123-ai/AVPENet/blob/main/avpenet/models/audio_encoder.py',
}


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix('.writing.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sync(target):
    if target.type == 'mps':
        torch.mps.synchronize()
    elif target.type == 'cuda':
        torch.cuda.synchronize(target)


def backbone(pretrained):
    net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    old = net.conv1
    net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    net.fc = nn.Identity()
    return net


def prefix(net, x):
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    return net.layer3(net.layer2(net.layer1(x)))


def embed(net, x):
    return net.avgpool(net.layer4(x)).flatten(1)


def log_mel(path):
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f'Invalid recording: {path}')
    duration = len(wave) / rate
    if rate != 16000:
        gcd = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // gcd, rate // gcd)
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    mel = librosa.feature.melspectrogram(y=wave, sr=16000, n_fft=400,
                                        hop_length=160, n_mels=128,
                                        fmin=50, fmax=8000)
    mel = np.log(mel + 1e-6)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    frames = mel.shape[1]
    mel = mel[:, :300]
    if frames < 300:
        mel = np.pad(mel, ((0, 0), (0, 300 - frames)))
    return torch.from_numpy(mel.astype(np.float32)).unsqueeze(0), duration, frames


def validate_manifest(manifest):
    sets = [set(manifest['splits'][s]) for s in ('train', 'validation', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Participant split overlap')
    seen = set()
    for e in manifest['examples']:
        if e['participant'] not in manifest['splits'][e['split']]:
            raise ValueError('Participant assigned to wrong split')
        if e['query']['clip'] in seen:
            raise ValueError('Duplicate query clip')
        seen.add(e['query']['clip'])
        if not 1 <= e['query']['score'] <= 10:
            raise ValueError('Invalid target score')
    return manifest['examples']


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = dict(SETTINGS)
    config.update(manifest_sha256=sha(args.manifest), code_sha256=sha(__file__))
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Run configuration changed; use a new output directory')
    else:
        write_json(config, out / 'config.json')
        (out / 'manifest.json').write_bytes(args.manifest.read_bytes())
    manifest = json.loads((out / 'manifest.json').read_text())
    examples = validate_manifest(manifest)
    target = device(args.device)
    provenance = {'config_sha256': sha(out / 'config.json'),
                  'manifest_sha256': sha(out / 'manifest.json')}
    if (out / 'features.pt').exists():
        checked(out / 'features.pt', provenance)
        return
    keys = [e['query']['clip'] for e in examples]
    if (out / 'mels.pt').exists():
        prep = checked(out / 'mels.pt', provenance)
        if prep['keys'] != keys:
            raise ValueError('Mel cache clip order differs')
        mels = prep['mels']
    else:
        files = {}
        for p in args.audio.rglob('*.wav'):
            if p.stem in files:
                raise ValueError(f'Duplicate audio filename: {p.stem}')
            files[p.stem] = p
        missing = set(keys) - files.keys()
        if missing:
            raise FileNotFoundError(f'Missing audio clips: {sorted(missing)[:5]}')
        durations, frame_counts, hashes, times, mel_list = [], [], {}, [], []
        for i, key in enumerate(keys):
            started = time.perf_counter()
            m, duration, frames = log_mel(files[key])
            times.append(time.perf_counter() - started)
            mel_list.append(m)
            durations.append(duration)
            frame_counts.append(frames)
            hashes[key] = sha(files[key])
            if i == 0 or (i + 1) % 100 == 0:
                print(f'Prepared {i + 1}/{len(keys)} clips', flush=True)
        mels = torch.stack(mel_list)
        atomic_save({**provenance, 'keys': keys, 'mels': mels}, out / 'mels.pt')
        write_json({'audio_sha256': hashes, 'duration_seconds': durations,
                    'clips_trimmed_to_300_frames': sum(x > 300 for x in frame_counts),
                    'clips_padded_to_300_frames': sum(x < 300 for x in frame_counts),
                    'preprocessing_seconds': sum(times),
                    'warm_preprocessing_median_ms': 1000 * statistics.median(times[1:]),
                    'split_counts': manifest['audit']['splits']}, out / 'data_audit.json')
    seed_all(SETTINGS['seed'])
    initial_path = out / 'encoder_initial.pt'
    if initial_path.exists():
        net = backbone(False)
        net.load_state_dict(load(initial_path))
    else:
        net = backbone(True)
        atomic_save(net.state_dict(), initial_path)
    net = net.to(target).eval().requires_grad_(False)
    feature_list, embedding_list = [], []
    sync(target)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(mels), 16):
            p = prefix(net, mels[start:start + 16].to(target))
            z = embed(net, p)
            if not torch.isfinite(z).all():
                raise ValueError('Non-finite embedding')
            feature_list.append(p.cpu())
            embedding_list.append(z.cpu())
            if start % 160 == 0:
                print(f'Encoded {min(start + 16, len(mels))}/{len(mels)} clips on {target}', flush=True)
    sync(target)
    elapsed = time.perf_counter() - started
    atomic_save({**provenance, 'keys': keys, 'prefix': torch.cat(feature_list),
                 'embeddings': torch.cat(embedding_list),
                 'initial_weights_sha256': sha(initial_path)}, out / 'features.pt')
    write_json({'device': str(target), 'torch': str(torch.__version__),
                'platform': platform.platform(), 'machine': platform.machine(),
                'packages': {p: importlib.metadata.version(p) for p in
                             ['torch', 'torchvision', 'librosa', 'numpy', 'scipy', 'soundfile']},
                'encoder_parameters': sum(p.numel() for p in net.parameters()),
                'extraction_batch_size': 16, 'extraction_seconds': elapsed,
                'clips': len(keys), 'initial_weights_sha256': sha(initial_path),
                'weights_url': ResNet34_Weights.IMAGENET1K_V1.url}, out / 'environment.json')


def checked(path, provenance):
    value = load(path)
    for k, v in provenance.items():
        if value[k] != v:
            raise ValueError(f'{path}: {k} mismatch')
    return value


class PredictionModel(nn.Module):
    def __init__(self, initial, mean, std, finetuned):
        super().__init__()
        self.finetuned = finetuned
        self.layer4 = copy.deepcopy(initial.layer4) if finetuned else nn.Identity()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(.2),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained batch statistics for this small participant dataset.
        for layer in self.layer4.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        z = self.layer4(x).mean(dim=(-2, -1)) if self.finetuned else x
        return self.head((z - self.mean) / self.std).squeeze(-1)


def predict(model, values, indices, target):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in indices.split(32):
            predictions.extend(model(values[batch].to(target)).cpu().tolist())
    return predictions


def resources(args):
    out = args.output
    config = json.loads((out / 'config.json').read_text())
    if config['code_sha256'] != sha(__file__):
        raise ValueError('Training code changed after protocol was fixed')
    provenance = {'config_sha256': sha(out / 'config.json'),
                  'manifest_sha256': sha(out / 'manifest.json')}
    examples = validate_manifest(json.loads((out / 'manifest.json').read_text()))
    cache = checked(out / 'features.pt', provenance)
    if cache['keys'] != [e['query']['clip'] for e in examples]:
        raise ValueError('Feature cache order differs')
    if cache['initial_weights_sha256'] != sha(out / 'encoder_initial.pt'):
        raise ValueError('Initial weights changed')
    initial = backbone(False)
    initial.load_state_dict(load(out / 'encoder_initial.pt'))
    indices = {s: torch.tensor([i for i, e in enumerate(examples) if e['split'] == s])
               for s in ('train', 'validation', 'test')}
    return examples, cache, initial, indices, provenance


def fit(args):
    target = device(args.device)
    examples, cache, initial, indices, provenance = resources(args)
    ti, vi = indices['train'], indices['validation']
    train_examples = [examples[i] for i in ti.tolist()]
    val_examples = [examples[i] for i in vi.tolist()]
    mean = cache['embeddings'][ti].mean(0)
    std = cache['embeddings'][ti].std(0).clamp_min(1e-4)
    weights = participant_weights(train_examples)
    weights /= weights.mean()
    labels = torch.tensor([e['query']['score'] for e in train_examples])
    for name in SETTINGS['variants']:
        checkpoint_path = args.output / f'{name}.pt'
        history_path = args.output / f'{name}_history.json'
        if history_path.exists():
            checked(checkpoint_path, provenance)
            continue
        if checkpoint_path.exists():
            raise FileExistsError(f'Interrupted training for {name}; review before restarting')
        seed_all(SETTINGS['seed'])
        model = PredictionModel(initial, mean, std, name == 'finetuned').to(target)
        values = cache['prefix'] if model.finetuned else cache['embeddings']
        groups = [{'params': model.head.parameters(), 'lr': SETTINGS['head_lr']}]
        if model.finetuned:
            groups.append({'params': model.layer4.parameters(), 'lr': SETTINGS['encoder_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=SETTINGS['weight_decay'])
        best, stale, history = float('inf'), 0, []
        started = time.perf_counter()
        for epoch in range(SETTINGS['max_epochs']):
            model.train()
            loss_sum = 0.
            for local_batch in torch.randperm(len(ti)).split(SETTINGS['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                pred = model(values[ti[local_batch]].to(target))
                loss = (nn.functional.huber_loss(pred, labels[local_batch].to(target),
                        reduction='none', delta=1.) * weights[local_batch].to(target)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(local_batch)
            predictions = predict(model, values, vi, target)
            score = metrics(val_examples, predictions)['participant_macro_mae']
            row = {'variant': name, 'epoch': epoch + 1, 'train_huber': loss_sum / len(ti),
                   'validation_mae': score, 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best:
                best, stale = score, 0
                atomic_save({**provenance, 'variant': name,
                             'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                             'best_epoch': epoch + 1, 'validation_mae': score,
                             'trainable_parameters': sum(p.numel() for p in model.parameters()),
                             'initial_weights_sha256': cache['initial_weights_sha256']}, checkpoint_path)
            else:
                stale += 1
            if stale >= SETTINGS['patience']:
                break
        write_json(history, history_path)
        del optimizer, model
    selections = {n: {k: load(args.output / f'{n}.pt')[k]
                      for k in ('best_epoch', 'validation_mae', 'trainable_parameters')}
                  for n in SETTINGS['variants']}
    write_json({'selected_variant': min(selections, key=lambda n: selections[n]['validation_mae']),
                'variants': selections}, args.output / 'selection.json')


def evaluate(args):
    report_path = args.output / 'test_results.json'
    if report_path.exists():
        raise FileExistsError('Test evaluation already saved')
    selection = json.loads((args.output / 'selection.json').read_text())
    examples, cache, initial, indices, provenance = resources(args)
    target = device(args.device)
    test_examples = [examples[i] for i in indices['test'].tolist()]
    results = {}
    checkpoint_hashes = {}
    for name in SETTINGS['variants']:
        ck = checked(args.output / f'{name}.pt', provenance)
        model = PredictionModel(initial, ck['state_dict']['mean'], ck['state_dict']['std'],
                                name == 'finetuned')
        model.load_state_dict(ck['state_dict'])
        model.to(target)
        values = cache['prefix'] if model.finetuned else cache['embeddings']
        results[name] = metrics(test_examples, predict(model, values, indices['test'], target))
        checkpoint_hashes[name] = sha(args.output / f'{name}.pt')
    train_examples = [examples[i] for i in indices['train'].tolist()]
    weights = participant_weights(train_examples)
    y = torch.tensor([e['query']['score'] for e in train_examples])
    mean = float((weights * y).sum() / weights.sum())
    order = y.argsort()
    median = float(y[order][torch.searchsorted(weights[order].cumsum(0), weights.sum() / 2)])
    for name, value in [('training_mean', mean), ('training_median', median)]:
        results[name] = metrics(test_examples, [value] * len(test_examples))
    results['reference_mean'] = metrics(test_examples,
        [float(np.mean([r['score'] for r in e['references']])) for e in test_examples])
    results['last_reference'] = metrics(test_examples,
        [e['references'][-1]['score'] for e in test_examples])
    people = sorted(results['frozen']['per_participant_mae'])
    differences = np.array([results['frozen']['per_participant_mae'][p] -
                            results['finetuned']['per_participant_mae'][p] for p in people])
    draws = np.random.default_rng(42).choice(differences, (10000, len(people)), replace=True).mean(1)
    report = {**provenance, 'selection': selection, 'checkpoint_sha256': checkpoint_hashes,
              'results': results, 'finetuning_mae_reduction': float(differences.mean()),
              'participant_bootstrap_95_ci': np.quantile(draws, [.025, .975]).tolist(),
              'test_participants': len(people), 'test_queries': len(test_examples)}
    write_json(report, report_path)
    print(json.dumps({k: v['participant_macro_mae'] for k, v in results.items()}, indent=2), flush=True)


def benchmark(args):
    # Fixed training examples; timing never influences model selection.
    target = device(args.device)
    examples, cache, initial, indices, provenance = resources(args)
    mels = checked(args.output / 'mels.pt', provenance)['mels'][indices['train'][:20]]
    output = {'device': str(target), 'batch_size': 1, 'warmup_calls': 5,
              'timed_calls': 100, 'input_shape': [1, 1, 128, 300],
              'scope': 'warm synchronized compute; input resident on device; '
                       'loading, decoding, preprocessing, and recording time separate', 'variants': {}}
    for name in SETTINGS['variants']:
        ck = checked(args.output / f'{name}.pt', provenance)
        net = copy.deepcopy(initial).to(target).eval()
        model = PredictionModel(initial, ck['state_dict']['mean'], ck['state_dict']['std'],
                                name == 'finetuned')
        model.load_state_dict(ck['state_dict'])
        model.to(target).eval()
        if name == 'finetuned':
            net.layer4.load_state_dict(model.layer4.state_dict())
        inputs = mels.to(target)
        times, full_times = [], []
        with torch.inference_mode():
            for i in range(105):
                x = inputs[i % len(inputs):i % len(inputs) + 1]
                sync(target)
                start = time.perf_counter()
                z = net(x)
                sync(target)
                encoded = time.perf_counter()
                model.head((z - model.mean) / model.std)
                sync(target)
                if i >= 5:
                    times.append((encoded - start) * 1000)
                    full_times.append((time.perf_counter() - start) * 1000)
        output['variants'][name] = {
            'encoder_median_ms': statistics.median(times),
            'encoder_p95_ms': float(np.quantile(times, .95)),
            'encoder_plus_head_median_ms': statistics.median(full_times),
            'encoder_plus_head_p95_ms': float(np.quantile(full_times, .95))}
    write_json(output, args.output / 'latency.json')
    print(json.dumps(output, indent=2), flush=True)


def export_model(args):
    selected = json.loads((args.output / 'selection.json').read_text())['selected_variant']
    ck = load(args.output / f'{selected}.pt')
    net = backbone(False)
    net.load_state_dict(load(args.output / 'encoder_initial.pt'))
    if selected == 'finetuned':
        net.layer4.load_state_dict({k.removeprefix('layer4.'): v
                                   for k, v in ck['state_dict'].items() if k.startswith('layer4.')})
    atomic_save({'encoder': net.state_dict(),
                 'head': {k.removeprefix('head.'): v for k, v in ck['state_dict'].items()
                          if k.startswith('head.')},
                 'mean': ck['state_dict']['mean'], 'std': ck['state_dict']['std'],
                 'selected_variant': selected, 'settings': SETTINGS,
                 'manifest_sha256': ck['manifest_sha256'],
                 'checkpoint_sha256': sha(args.output / f'{selected}.pt')},
                args.output / 'pain_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'fit', 'evaluate', 'benchmark', 'export_model', 'all'])
    parser.add_argument('--manifest', type=Path, default=Path('runs/ast_reference/manifest.json'))
    parser.add_argument('--audio', type=Path, default=Path('data/tame/mic1_trim_v1'))
    parser.add_argument('--output', type=Path, default=Path('runs/resnet_comparison'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(42)
    stages = [prepare, fit, evaluate, benchmark, export_model] if args.stage == 'all' else [globals()[args.stage]]
    for stage in stages:
        stage(args)
