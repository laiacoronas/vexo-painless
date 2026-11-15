"""Compare normalized kernel ridge and PLS on a fixed personal calibration split."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from train import load, sha


def unit_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('Embedding norms must be positive.')
    return x / norm


def distances(x, y):
    return np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None] - 2 * x @ y.T, 0.)


def fit_kernel(x, y, alpha, bandwidth_multiplier):
    z = unit_rows(x)
    d = distances(z, z)
    median_distance = float(np.median(d[np.triu_indices(len(z), 1)]))
    if median_distance <= 0:
        raise ValueError('Training embeddings must have positive median distance.')
    gamma = 1 / (median_distance * bandwidth_multiplier ** 2)
    kernel = np.exp(-gamma * d)
    column_mean, grand_mean = kernel.mean(axis=0), kernel.mean()
    centered = kernel - column_mean[None] - column_mean[:, None] + grand_mean
    offset = float(y.mean())
    dual = np.linalg.solve(centered + alpha * np.eye(len(z)), y - offset)
    return dict(kind=np.array('kernel_ridge'), training_embeddings=z, gamma=np.array(gamma),
                column_mean=column_mean, grand_mean=np.array(grand_mean), dual=dual,
                offset=np.array(offset), alpha=np.array(alpha),
                bandwidth_multiplier=np.array(bandwidth_multiplier),
                median_squared_distance=np.array(median_distance))


def fit_pls(x, y, components):
    model = PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-8)
    model.fit(x, y)
    checkpoint = dict(kind=np.array('pls'), mean=model._x_mean.copy(),
                      coefficient=model.coef_.reshape(-1).copy(),
                      offset=np.array(float(model.intercept_.item())),
                      components=np.array(components),
                      x_weights=model.x_weights_.copy(), x_loadings=model.x_loadings_.copy())
    np.testing.assert_allclose(predict(checkpoint, x), model.predict(x).reshape(-1), atol=1e-10, rtol=1e-10)
    return checkpoint


def predict(checkpoint, x):
    if str(checkpoint['kind']) == 'pls':
        return (x - checkpoint['mean']) @ checkpoint['coefficient'] + checkpoint['offset']
    kernel = np.exp(-checkpoint['gamma'] * distances(unit_rows(x), checkpoint['training_embeddings']))
    centered = kernel - kernel.mean(axis=1, keepdims=True) - checkpoint['column_mean'][None] + checkpoint['grand_mean']
    return centered @ checkpoint['dual'] + checkpoint['offset']


def fit_candidate(name, settings, x, y):
    return (fit_kernel(x, y, settings['alpha'], settings['bandwidth_multiplier'])
            if name == 'kernel_ridge' else fit_pls(x, y, settings['components']))


def select(x, y, calibration, grids):
    takes = np.array([r['take'] for r in calibration])
    assert set(takes) == {1, 2}
    candidates = {}
    for name, settings_grid in grids.items():
        candidates[name] = []
        for settings in settings_grid:
            oof = np.empty(len(y))
            fold_mae = []
            for take in (1, 2):
                train, validation = takes == take, takes != take
                checkpoint = fit_candidate(name, settings, x[train], y[train])
                oof[validation] = predict(checkpoint, x[validation])
                fold_mae.append(float(np.abs(oof[validation] - y[validation]).mean()))
            candidates[name].append(dict(settings=settings, fold_mae=fold_mae,
                                         mean_mae=float(np.mean(fold_mae)),
                                         calibration_oof_predictions=oof.tolist()))
        print(f'Completed {name}: {len(settings_grid)} candidates, two calibration folds each', flush=True)
    chosen = {
        'kernel_ridge': min(candidates['kernel_ridge'], key=lambda r: (r['mean_mae'], -r['settings']['alpha'], -r['settings']['bandwidth_multiplier'])),
        'pls': min(candidates['pls'], key=lambda r: (r['mean_mae'], r['settings']['components'])),
    }
    return dict(selected=chosen, candidates=candidates)


def summarize(predictions, target):
    error = predictions - target
    return dict(validation_mae=float(np.abs(error).mean()),
                validation_rmse=float(np.sqrt(np.mean(error ** 2))),
                validation_pain_level_macro_mae=float(np.mean([
                    np.abs(error[target == level]).mean() for level in sorted(set(target))])),
                predictions_outside_0_10=int(np.sum((predictions < 0) | (predictions > 10))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/milan_pain/personal_calibration'))
    parser.add_argument('--embeddings', type=Path, default=Path('runs/milan_pain/reference_0_2_5_remaining/embeddings.pt'))
    parser.add_argument('--labels', type=Path, default=Path('data/milan_pain/labels.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    previous_plan = json.loads((args.baseline / 'plan.json').read_text())
    cache = load(args.embeddings)
    assert previous_plan['embeddings_sha256'] == sha(args.embeddings)
    assert previous_plan['labels_sha256'] == sha(args.labels)
    calibration, validation = previous_plan['calibration'], previous_plan['validation']
    assert len(calibration) == 16 and len(validation) == 11
    assert {r['file'] for r in calibration}.isdisjoint({r['file'] for r in validation})
    for row in calibration + validation:
        assert cache['audio_sha256'][row['file']] == sha(args.labels.parent / row['file'])
    grids = dict(kernel_ridge=[dict(alpha=a, bandwidth_multiplier=b)
                              for a in [0.001, 0.01, 0.1, 1., 10.]
                              for b in [0.25, 0.5, 1., 2., 4.]],
                 pls=[dict(components=n) for n in [1, 2, 3, 4]])
    plan = dict(calibration=calibration, validation=validation, grids=grids,
                tuning_policy=previous_plan['tuning_policy'],
                kernel_policy='Unit-length audio embeddings; RBF bandwidth from training-fold median squared distance times multiplier squared; centered kernel and unpenalized target-mean intercept.',
                pls_policy='PLSRegression with training-fold centering and variance scaling; select one to four components.',
                evaluation_scope='Exploratory comparison on the previously inspected within-recording validation split; calibration and validation can share rating blocks.',
                output_policy='Raw predictions on the reported 0–10 scale.',
                encoder_model=cache['encoder_model'], encoder_revision=cache['encoder_revision'],
                baseline_plan_sha256=sha(args.baseline / 'plan.json'),
                embeddings_sha256=sha(args.embeddings), labels_sha256=sha(args.labels),
                code_sha256=sha(__file__), numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    x = np.stack([cache['embeddings'][r['file']].numpy() for r in calibration]).astype(np.float64)
    y = np.array([r['pain_level'] for r in calibration], dtype=np.float64)
    selection = select(x, y, calibration, grids)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    models = {}
    for name, chosen in selection['selected'].items():
        models[name] = fit_candidate(name, chosen['settings'], x, y)
        models[name].update(encoder_model=np.array(cache['encoder_model']),
                            encoder_revision=np.array(cache['encoder_revision']),
                            plan_sha256=np.array(sha(args.output / 'plan.json')))
        np.savez(args.output / f'{name}.npz', **models[name])
    # Both selected models are fixed and saved before validation predictions.
    vx = np.stack([cache['embeddings'][r['file']].numpy() for r in validation]).astype(np.float64)
    vy = np.array([r['pain_level'] for r in validation], dtype=np.float64)
    baseline_results = json.loads((args.baseline / 'validation_results.json').read_text())
    assert [r['file'] for r in baseline_results['recordings']] == [r['file'] for r in validation]
    rows = [dict(r) for r in baseline_results['recordings']]
    metrics = dict(baseline_results['metrics'])
    for name, model in models.items():
        pred = predict(model, vx)
        assert np.isfinite(pred).all()
        metrics[name] = dict(summarize(pred, vy), calibration_mae=float(np.abs(predict(model, x) - y).mean()),
                             calibration_cv_mae=selection['selected'][name]['mean_mae'])
        for row, value in zip(rows, pred):
            row[f'{name}_prediction'] = float(value)
            row[f'{name}_absolute_error'] = float(abs(value - row['pain_level']))
    by_level = []
    for level in sorted(set(vy)):
        subset = [r for r in rows if r['pain_level'] == level]
        by_level.append(dict(reported_pain=int(level), count=len(subset), **{
            f'{name}_{suffix}': float(np.mean([r[f'{name}_{key}'] for r in subset]))
            for name in metrics for suffix, key in [('mean_prediction', 'prediction'), ('mae', 'absolute_error')]}))
    report = dict(calibration_count=16, validation_count=11, metrics=metrics, recordings=rows,
                  by_pain_level=by_level, selected_settings={n: r['settings'] for n, r in selection['selected'].items()},
                  plan_sha256=sha(args.output / 'plan.json'),
                  checkpoint_sha256={n: sha(args.output / f'{n}.npz') for n in models})
    (args.output / 'validation_results.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(metrics=metrics, selected_settings=report['selected_settings'], by_pain_level=by_level), indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
