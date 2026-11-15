"""Calibrate frozen AST embeddings with a small personal MLP and ridge head."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train import load, sha


class PersonalMLP(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def scale_fit(x, y):
    return dict(mean=x.mean(0), std=x.std(0, unbiased=False).clamp_min(1e-4), offset=y.mean())


def scale(x, stats):
    return (x - stats['mean']) / stats['std']


def ridge_fit(x, y, alpha):
    stats = scale_fit(x, y)
    z = scale(x, stats).double()
    centered = (y - stats['offset']).double()
    weight = z.T @ torch.linalg.solve(z @ z.T + alpha * torch.eye(len(z), dtype=z.dtype), centered)
    return dict(kind='ridge', alpha=alpha, weight=weight, **stats)


def predict(checkpoint, x):
    z = scale(x, checkpoint)
    if checkpoint['kind'] == 'ridge':
        return z.double() @ checkpoint['weight'] + checkpoint['offset']
    model = PersonalMLP(x.shape[1]).eval()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.inference_mode():
        return model(z) + checkpoint['offset']


def mlp_fit(x, y, weight_decay, epochs, probe=None, probe_epochs=()):
    torch.manual_seed(42)
    stats = scale_fit(x, y)
    z, target = scale(x, stats), y - stats['offset']
    model = PersonalMLP(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    probes, history = {}, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(z), target)
        if not torch.isfinite(loss):
            raise ValueError('Training loss must be finite.')
        loss.backward()
        optimizer.step()
        if epoch in probe_epochs or epoch == epochs:
            history.append(dict(epoch=epoch, training_mse_before_step=loss.item()))
        if probe is not None and epoch in probe_epochs:
            model.eval()
            with torch.inference_mode():
                probes[epoch] = (model(scale(probe, stats)) + stats['offset']).clone()
    return dict(kind='mlp', state_dict=model.state_dict(), weight_decay=weight_decay,
                epochs=epochs, learning_rate=1e-3, seed=42, architecture=[x.shape[1], 16, 1],
                activation='GELU', objective='MSE', **stats), probes, history


def select_settings(x, y, records, alphas, decays, epochs):
    folds = [(torch.tensor([r['take'] == train_take for r in records]),
              torch.tensor([r['take'] != train_take for r in records])) for train_take in (1, 2)]
    ridge_scores = {a: [] for a in alphas}
    mlp_scores = {(d, e): [] for d in decays for e in epochs}
    for fold, (train, validation) in enumerate(folds, 1):
        for alpha in alphas:
            pred = predict(ridge_fit(x[train], y[train], alpha), x[validation])
            ridge_scores[alpha].append(float((pred - y[validation]).abs().mean()))
        for decay in decays:
            _, probes, _ = mlp_fit(x[train], y[train], decay, max(epochs), x[validation], epochs)
            for epoch, pred in probes.items():
                mlp_scores[(decay, epoch)].append(float((pred - y[validation]).abs().mean()))
        print(f'Completed calibration fold {fold}/2', flush=True)
    ridge = [dict(alpha=a, fold_mae=v, mean_mae=float(np.mean(v))) for a, v in ridge_scores.items()]
    mlp = [dict(weight_decay=d, epochs=e, fold_mae=v, mean_mae=float(np.mean(v)))
           for (d, e), v in mlp_scores.items()]
    return dict(ridge=min(ridge, key=lambda r: (r['mean_mae'], -r['alpha'])),
                mlp=min(mlp, key=lambda r: (r['mean_mae'], -r['weight_decay'], r['epochs'])),
                ridge_candidates=ridge, mlp_candidates=mlp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=Path('data/milan_pain/labels.json'))
    parser.add_argument('--embeddings', type=Path, default=Path('runs/milan_pain/reference_0_2_5_remaining/embeddings.pt'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    labels = json.loads(args.labels.read_text())
    cache = load(args.embeddings)
    grouped = {}
    for row in labels['recordings']:
        grouped.setdefault(row['pain_level'], []).append(row)
        assert cache['audio_sha256'][row['file']] == sha(args.labels.parent / row['file'])
        assert torch.isfinite(cache['embeddings'][row['file']]).all()
    calibration, validation = [], []
    for level, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['source_start_seconds'])
        if len(rows) < 3:
            raise ValueError(f'Pain {level} requires at least three phrases.')
        calibration.extend(rows[:2])
        validation.extend(rows[2:])
    assert len(calibration) == 16 and len(validation) == 11
    assert {r['take'] for r in calibration} == {1, 2}
    assert {r['file'] for r in calibration}.isdisjoint({r['file'] for r in validation})
    alphas, decays, epochs = [0.1, 1., 10., 100., 1000., 10000.], [0.01, 0.1, 1.], [25, 50, 100, 200, 400]
    plan = dict(calibration=calibration, validation=validation,
                calibration_policy='First two phrases per reported pain level, ordered by source time.',
                validation_policy='Every remaining phrase; four at pain 8 and one at each other level.',
                tuning_policy='Two folds within calibration: take 1 trains/take 2 validates, then swap. Select lowest mean MAE. Fit scalers and target centering using only the training fold.',
                evaluation_scope='Within-person, within-recording held-out phrases. Calibration and validation can share a reported-rating block. Segmentation and earlier model predictions were inspected before this experiment.',
                encoder_model=cache['encoder_model'], encoder_revision=cache['encoder_revision'],
                mlp=dict(architecture=[768, 16, 1], activation='GELU', optimizer='AdamW',
                         learning_rate=1e-3, objective='MSE', seed=42, batch='full calibration set',
                         weight_decay_candidates=decays, epoch_candidates=epochs),
                ridge=dict(objective='sum squared errors + alpha * squared weight norm; unpenalized intercept', alpha_candidates=alphas),
                output_policy='Raw predictions on the reported 0–10 scale.',
                labels_sha256=sha(args.labels), embeddings_sha256=sha(args.embeddings),
                training_code_sha256=sha(__file__), torch_version=str(torch.__version__))
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    x = torch.stack([cache['embeddings'][r['file']] for r in calibration])
    y = torch.tensor([float(r['pain_level']) for r in calibration])
    selection = select_settings(x, y, calibration, alphas, decays, epochs)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    models = {'ridge': ridge_fit(x, y, selection['ridge']['alpha'])}
    models['mlp'], _, history = mlp_fit(x, y, selection['mlp']['weight_decay'], selection['mlp']['epochs'])
    (args.output / 'mlp_history.json').write_text(json.dumps(history, indent=2))
    for name, checkpoint in models.items():
        checkpoint.update(encoder_model=cache['encoder_model'], encoder_revision=cache['encoder_revision'],
                          plan_sha256=sha(args.output / 'plan.json'))
        torch.save(checkpoint, args.output / f'{name}.pt')
    # Held-out embeddings and targets enter only after settings and weights are saved.
    vx = torch.stack([cache['embeddings'][r['file']] for r in validation])
    vy = torch.tensor([float(r['pain_level']) for r in validation])
    rows = [dict(r) for r in validation]
    metrics = {}
    for name, checkpoint in models.items():
        pred = predict(checkpoint, vx)
        assert torch.isfinite(pred).all()
        calibration_pred = predict(checkpoint, x)
        error = pred - vy
        metrics[name] = dict(validation_mae=float(error.abs().mean()),
                             validation_rmse=float(error.square().mean().sqrt()),
                             validation_pain_level_macro_mae=float(np.mean([
                                 float(error[vy == level].abs().mean()) for level in sorted(grouped)])),
                             calibration_mae=float((calibration_pred - y).abs().mean()),
                             calibration_cv_mae=selection[name]['mean_mae'],
                             predictions_outside_0_10=int(((pred < 0) | (pred > 10)).sum()))
        for row, value in zip(rows, pred.tolist()):
            row[f'{name}_prediction'] = value
            row[f'{name}_absolute_error'] = abs(value - row['pain_level'])
    by_level = []
    for level in sorted(grouped):
        subset = [r for r in rows if r['pain_level'] == level]
        by_level.append(dict(reported_pain=level, count=len(subset), **{
            f'{name}_{suffix}': float(np.mean([r[f'{name}_{key}'] for r in subset]))
            for name in models for suffix, key in [('mean_prediction', 'prediction'), ('mae', 'absolute_error')]}))
    report = dict(calibration_count=len(calibration), validation_count=len(validation),
                  metrics=metrics, by_pain_level=by_level, recordings=rows,
                  selected_settings={name: selection[name] for name in models},
                  checkpoint_sha256={name: sha(args.output / f'{name}.pt') for name in models},
                  plan_sha256=sha(args.output / 'plan.json'))
    (args.output / 'validation_results.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(metrics=metrics, selected_settings=report['selected_settings'], by_pain_level=by_level), indent=2), flush=True)


if __name__ == '__main__':
    main()
