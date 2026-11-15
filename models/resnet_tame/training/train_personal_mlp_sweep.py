"""Compare four personal MLP architectures on the saved calibration split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train import load, sha
from train_personal import scale, scale_fit


class MLP(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(architecture, architecture[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(architecture) - 2:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit(x, y, architecture, decay, epochs, probe=None, probe_epochs=()):
    torch.manual_seed(42)
    stats = scale_fit(x, y)
    z, target = scale(x, stats), y - stats['offset']
    model = MLP(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
    probes = {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(z), target)
        if not torch.isfinite(loss):
            raise ValueError('Training loss must be finite.')
        loss.backward()
        optimizer.step()
        if probe is not None and epoch in probe_epochs:
            model.eval()
            with torch.inference_mode():
                probes[epoch] = (model(scale(probe, stats)) + stats['offset']).clone()
    return dict(kind='mlp', architecture=architecture, state_dict=model.state_dict(),
                parameter_count=sum(p.numel() for p in model.parameters()),
                weight_decay=decay, epochs=epochs, learning_rate=1e-3, seed=42,
                activation='GELU', objective='MSE', **stats), probes


def predict(checkpoint, x):
    model = MLP(checkpoint['architecture']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.inference_mode():
        return model(scale(x, checkpoint)) + checkpoint['offset']


def select(x, y, calibration, architectures, decays, epochs):
    takes = torch.tensor([r['take'] for r in calibration])
    candidates, chosen = {}, {}
    for name, architecture in architectures.items():
        scores = {(d, e): [] for d in decays for e in epochs}
        oof = {(d, e): torch.empty_like(y) for d in decays for e in epochs}
        for take in (1, 2):
            train, validation = takes == take, takes != take
            for decay in decays:
                _, probes = fit(x[train], y[train], architecture, decay, max(epochs), x[validation], epochs)
                for epoch, prediction in probes.items():
                    scores[(decay, epoch)].append(float((prediction - y[validation]).abs().mean()))
                    oof[(decay, epoch)][validation] = prediction
        candidates[name] = [dict(weight_decay=d, epochs=e, fold_mae=s, mean_mae=float(np.mean(s)),
                                 calibration_oof_predictions=oof[(d, e)].tolist())
                            for (d, e), s in scores.items()]
        chosen[name] = min(candidates[name], key=lambda r: (r['mean_mae'], -r['weight_decay'], r['epochs']))
        print(f"{name}: calibration CV MAE {chosen[name]['mean_mae']:.4f}; "
              f"weight decay {chosen[name]['weight_decay']}, epochs {chosen[name]['epochs']}", flush=True)
    return dict(selected=chosen, candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/milan_pain/personal_calibration'))
    parser.add_argument('--comparison', type=Path, default=Path('runs/milan_pain/personal_alternatives'))
    parser.add_argument('--embeddings', type=Path, default=Path('runs/milan_pain/reference_0_2_5_remaining/embeddings.pt'))
    parser.add_argument('--labels', type=Path, default=Path('data/milan_pain/labels.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    previous_plan = json.loads((args.baseline / 'plan.json').read_text())
    assert previous_plan['embeddings_sha256'] == sha(args.embeddings)
    assert previous_plan['labels_sha256'] == sha(args.labels)
    cache = load(args.embeddings)
    calibration, validation = previous_plan['calibration'], previous_plan['validation']
    assert len(calibration) == 16 and len(validation) == 11
    assert {r['file'] for r in calibration}.isdisjoint({r['file'] for r in validation})
    for row in calibration + validation:
        assert sha(args.labels.parent / row['file']) == cache['audio_sha256'][row['file']]
    architectures = {'mlp_4': [768, 4, 1], 'mlp_8': [768, 8, 1],
                     'mlp_32': [768, 32, 1], 'mlp_32_8': [768, 32, 8, 1]}
    decays, epochs = [0.01, 0.1, 1.], [25, 50, 100, 200, 400]
    plan = dict(calibration=calibration, validation=validation, architectures=architectures,
                weight_decay_candidates=decays, epoch_candidates=epochs,
                tuning_policy=previous_plan['tuning_policy'], activation='GELU', optimizer='AdamW',
                learning_rate=1e-3, seed=42, objective='MSE', batch='full calibration set',
                evaluation_scope='Exploratory comparison on the reused within-recording validation split; calibration and validation can share rating blocks.',
                output_policy='Raw predictions on the reported 0–10 scale.',
                encoder_model=cache['encoder_model'], encoder_revision=cache['encoder_revision'],
                baseline_plan_sha256=sha(args.baseline / 'plan.json'), labels_sha256=sha(args.labels),
                embeddings_sha256=sha(args.embeddings), code_sha256=sha(__file__),
                scaling_code_sha256=sha('train_personal.py'), torch_version=str(torch.__version__))
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    x = torch.stack([cache['embeddings'][r['file']] for r in calibration])
    y = torch.tensor([float(r['pain_level']) for r in calibration])
    selection = select(x, y, calibration, architectures, decays, epochs)
    selection['preferred_new_architecture_by_calibration_cv'] = min(
        selection['selected'], key=lambda name: selection['selected'][name]['mean_mae'])
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    models = {}
    for name, architecture in architectures.items():
        settings = selection['selected'][name]
        models[name], _ = fit(x, y, architecture, settings['weight_decay'], settings['epochs'])
        models[name].update(encoder_model=cache['encoder_model'], encoder_revision=cache['encoder_revision'],
                            plan_sha256=sha(args.output / 'plan.json'))
        torch.save(models[name], args.output / f'{name}.pt')
    # Evaluate after every model's settings and weights have been saved.
    vx = torch.stack([cache['embeddings'][r['file']] for r in validation])
    vy = torch.tensor([float(r['pain_level']) for r in validation])
    previous = json.loads((args.comparison / 'validation_results.json').read_text())
    assert [r['file'] for r in previous['recordings']] == [r['file'] for r in validation]
    rows = [dict(r) for r in previous['recordings']]
    metrics = dict(previous['metrics'])
    for name, checkpoint in models.items():
        pred = predict(checkpoint, vx)
        assert torch.isfinite(pred).all()
        error = pred - vy
        metrics[name] = dict(validation_mae=float(error.abs().mean()),
                             validation_rmse=float(error.square().mean().sqrt()),
                             validation_pain_level_macro_mae=float(np.mean([
                                 float(error[vy == level].abs().mean()) for level in sorted(set(vy.tolist()))])),
                             calibration_mae=float((predict(checkpoint, x) - y).abs().mean()),
                             calibration_cv_mae=selection['selected'][name]['mean_mae'],
                             predictions_outside_0_10=int(((pred < 0) | (pred > 10)).sum()))
        for row, value in zip(rows, pred.tolist()):
            row[f'{name}_prediction'] = value
            row[f'{name}_absolute_error'] = abs(value - row['pain_level'])
    by_level = []
    for level in sorted(set(vy.tolist())):
        subset = [r for r in rows if r['pain_level'] == level]
        by_level.append(dict(reported_pain=int(level), count=len(subset), **{
            f'{name}_{suffix}': float(np.mean([r[f'{name}_{key}'] for r in subset]))
            for name in metrics for suffix, key in [('mean_prediction', 'prediction'), ('mae', 'absolute_error')]}))
    report = dict(calibration_count=16, validation_count=11, metrics=metrics, by_pain_level=by_level,
                  recordings=rows, selected_settings=selection['selected'],
                  preferred_new_architecture_by_calibration_cv=selection['preferred_new_architecture_by_calibration_cv'],
                  architectures=architectures,
                  parameter_counts={name: c['parameter_count'] for name, c in models.items()},
                  checkpoint_sha256={name: sha(args.output / f'{name}.pt') for name in models},
                  plan_sha256=sha(args.output / 'plan.json'))
    (args.output / 'validation_results.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(metrics=metrics, parameter_counts=report['parameter_counts'],
                          preferred_new_architecture_by_calibration_cv=report['preferred_new_architecture_by_calibration_cv']), indent=2), flush=True)


if __name__ == '__main__':
    main()
