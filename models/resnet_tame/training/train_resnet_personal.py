"""Calibrate five MLPs and kernel ridge on frozen TAME-trained ResNet features."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from resnet_experiment import backbone, log_mel
from train import load, sha
from train_personal_alternatives import fit_kernel, predict as predict_kernel, summarize
from train_personal_mlp_sweep import fit as fit_mlp, predict as predict_mlp, select as select_mlp


def select_kernel(x, y, records, settings_grid):
    takes = np.array([r['take'] for r in records])
    candidates = []
    for settings in settings_grid:
        oof, scores = np.empty(len(y)), []
        for take in (1, 2):
            tr, va = takes == take, takes != take
            checkpoint = fit_kernel(x[tr], y[tr], **settings)
            oof[va] = predict_kernel(checkpoint, x[va])
            scores.append(float(np.abs(oof[va] - y[va]).mean()))
        candidates.append(dict(settings=settings, fold_mae=scores, mean_mae=float(np.mean(scores)),
                               calibration_oof_predictions=oof.tolist()))
    best = min(candidates, key=lambda r: (r['mean_mae'], -r['settings']['alpha'], -r['settings']['bandwidth_multiplier']))
    print(f"Kernel ridge: calibration CV MAE {best['mean_mae']:.4f}", flush=True)
    return dict(selected=best, candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/resnet_comparison/pain_model.pt'))
    parser.add_argument('--labels', type=Path, default=Path('data/milan_pain/labels.json'))
    parser.add_argument('--baseline-plan', type=Path, default=Path('runs/milan_pain/personal_calibration/plan.json'))
    parser.add_argument('--ast-results', type=Path, default=Path('runs/milan_pain/personal_mlp_sweep/validation_results.json'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    prior = json.loads(args.baseline_plan.read_text())
    labels = json.loads(args.labels.read_text())
    assert prior['labels_sha256'] == sha(args.labels)
    calibration, validation = prior['calibration'], prior['validation']
    assert len(calibration) == 16 and len(validation) == 11
    assert {r['file'] for r in calibration}.isdisjoint({r['file'] for r in validation})
    assert {r['file'] for r in calibration + validation} == {r['file'] for r in labels['recordings']}
    bundle = load(args.checkpoint)
    assert bundle['selected_variant'] == 'finetuned'
    architectures = {f'mlp_{w}': [512, w, 1] for w in [4, 8, 16, 32]}
    architectures['mlp_32_8'] = [512, 32, 8, 1]
    decays, epochs = [0.01, 0.1, 1.], [25, 50, 100, 200, 400]
    kernel_grid = [dict(alpha=a, bandwidth_multiplier=b) for a in [0.001, 0.01, 0.1, 1., 10.] for b in [0.25, 0.5, 1., 2., 4.]]
    plan = dict(calibration=calibration, validation=validation, architectures=architectures,
                encoder='TAME-trained single-channel ResNet-34', encoder_frozen=True, embedding_width=512,
                encoder_checkpoint=str(args.checkpoint), encoder_checkpoint_sha256=sha(args.checkpoint),
                selected_encoder_variant=bundle['selected_variant'], preprocessing=bundle['settings'],
                activation='GELU', optimizer='AdamW', learning_rate=1e-3, seed=42, objective='MSE',
                mlp_weight_decay_candidates=decays, mlp_epoch_candidates=epochs, kernel_grid=kernel_grid,
                tuning_policy=prior['tuning_policy'],
                kernel_policy='Unit-normalized embeddings; centered RBF kernel; fold-training median-distance bandwidth; unpenalized target-mean intercept.',
                output_policy='Raw predictions on the reported 0–10 scale.',
                evaluation_scope='Exploratory comparison on the reused within-recording split; calibration and evaluation phrases can share rating blocks.',
                baseline_plan_sha256=sha(args.baseline_plan), labels_sha256=sha(args.labels),
                source_code_sha256={name: sha(name) for name in [__file__, 'resnet_experiment.py', 'train_personal_mlp_sweep.py', 'train_personal_alternatives.py', 'train_personal.py']},
                packages={name: importlib.metadata.version(name) for name in ['torch','torchvision','numpy','scipy','librosa','scikit-learn']})
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    encoder = backbone(False)
    encoder.load_state_dict(bundle['encoder'])
    encoder.eval().requires_grad_(False)
    embeddings, hashes, audit = {}, {}, []
    with torch.inference_mode():
        for i, row in enumerate(labels['recordings'], 1):
            path = args.labels.parent / row['file']
            mel, duration, frames = log_mel(path)
            embedding = encoder(mel[None])[0]
            assert embedding.shape == (512,) and torch.isfinite(embedding).all()
            embeddings[row['file']] = embedding.clone()
            hashes[row['file']] = sha(path)
            audit.append(dict(file=row['file'], duration_seconds=duration, mel_frames_available=frames,
                              padded_frames=max(0, 300-frames), truncated_frames=max(0, frames-300)))
            print(f'Embedded {i}/27: {path.name}', flush=True)
    assert all(torch.equal(v, bundle['encoder'][name]) for name, v in encoder.state_dict().items())
    torch.save(dict(embeddings=embeddings, audio_sha256=hashes,
                    encoder_checkpoint_sha256=plan['encoder_checkpoint_sha256'],
                    plan_sha256=sha(args.output / 'plan.json')), args.output / 'embeddings.pt')
    (args.output / 'extraction_audit.json').write_text(json.dumps(dict(recordings=audit, encoder_weights_unchanged=True, device='cpu'), indent=2))
    torch.set_num_threads(1)
    x = torch.stack([embeddings[r['file']] for r in calibration])
    y = torch.tensor([float(r['pain_level']) for r in calibration])
    xd, yd = x.numpy().astype(np.float64), y.numpy().astype(np.float64)
    mlp_selection = select_mlp(x, y, calibration, architectures, decays, epochs)
    kernel_selection = select_kernel(xd, yd, calibration, kernel_grid)
    selection = dict(mlp=mlp_selection, kernel_ridge=kernel_selection)
    all_scores = {name: r['mean_mae'] for name, r in mlp_selection['selected'].items()}
    all_scores['kernel_ridge'] = kernel_selection['selected']['mean_mae']
    selection['preferred_model_by_calibration_cv'] = min(all_scores, key=all_scores.get)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    models = {}
    for name, architecture in architectures.items():
        settings = mlp_selection['selected'][name]
        models[name], _ = fit_mlp(x, y, architecture, settings['weight_decay'], settings['epochs'])
        models[name].update(encoder_checkpoint_sha256=plan['encoder_checkpoint_sha256'], plan_sha256=sha(args.output / 'plan.json'))
        torch.save(models[name], args.output / f'{name}.pt')
    kernel = fit_kernel(xd, yd, **kernel_selection['selected']['settings'])
    kernel.update(encoder_checkpoint_sha256=np.array(plan['encoder_checkpoint_sha256']), plan_sha256=np.array(sha(args.output / 'plan.json')))
    np.savez(args.output / 'kernel_ridge.npz', **kernel)
    # Evaluate after all calibration choices and trained weights are saved.
    vx = torch.stack([embeddings[r['file']] for r in validation])
    vy = np.array([r['pain_level'] for r in validation], dtype=float)
    ast = json.loads(args.ast_results.read_text())
    assert [r['file'] for r in ast['recordings']] == [r['file'] for r in validation]
    rows = [dict(r) for r in validation]
    metrics, comparison = {}, {}
    for name in [*models, 'kernel_ridge']:
        if name == 'kernel_ridge':
            p, cp = predict_kernel(kernel, vx.numpy().astype(float)), predict_kernel(kernel, xd)
        else:
            p, cp = predict_mlp(models[name], vx).numpy(), predict_mlp(models[name], x).numpy()
        assert np.isfinite(p).all()
        metrics[name] = dict(summarize(p, vy), calibration_mae=float(np.abs(cp-yd).mean()), calibration_cv_mae=all_scores[name])
        ast_name = 'mlp' if name == 'mlp_16' else name
        comparison[name] = dict(ast_validation_mae=ast['metrics'][ast_name]['validation_mae'], resnet_validation_mae=metrics[name]['validation_mae'],
                                resnet_mae_reduction=ast['metrics'][ast_name]['validation_mae']-metrics[name]['validation_mae'])
        for i, (row, value) in enumerate(zip(rows, p)):
            row[f'resnet_{name}_prediction'] = float(value)
            row[f'resnet_{name}_absolute_error'] = float(abs(value-row['pain_level']))
            row[f'ast_{name}_prediction'] = ast['recordings'][i][f'{ast_name}_prediction']
    report = dict(calibration_count=16, validation_count=11, metrics=metrics, comparison=comparison,
                  recordings=rows, preferred_model_by_calibration_cv=selection['preferred_model_by_calibration_cv'],
                  parameter_counts={name: c['parameter_count'] for name, c in models.items()},
                  checkpoint_sha256={name: sha(args.output / f'{name}.pt') for name in models},
                  kernel_checkpoint_sha256=sha(args.output / 'kernel_ridge.npz'),
                  ast_results_sha256=sha(args.ast_results), plan_sha256=sha(args.output / 'plan.json'))
    (args.output / 'validation_results.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(metrics=metrics, comparison=comparison, preferred_model_by_calibration_cv=report['preferred_model_by_calibration_cv']), indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
