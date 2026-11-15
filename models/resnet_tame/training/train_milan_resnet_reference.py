"""Lamita-style per-epoch personal calibration, with and without pain-0 references."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train import load, sha
from train_personal import scale, scale_fit
from train_personal_mlp_sweep import MLP


def arrays(records, embeddings):
    return (torch.stack([embeddings[r['file']] for r in records]),
            torch.tensor([float(r['pain_level']) for r in records]))


def inputs(records, embeddings, stats, references):
    q = scale(arrays(records, embeddings)[0], stats)
    if not references:
        return q, None
    r = scale(arrays(references, embeddings)[0], stats)
    qe = q[:, None].expand(-1, len(r), -1)
    re = r[None].expand(len(q), -1, -1)
    mask = torch.tensor([[q['file'] != r['file'] for r in references] for q in records])
    assert (mask.sum(1) > 0).all()
    return torch.cat([qe, re, qe-re], dim=-1), mask


def forward(model, features, mask, offset):
    value = model(features)
    if mask is not None:
        value = (value*mask).sum(1)/mask.sum(1)
    return value+offset


def train_run(records, embeddings, reference_mode, decay, epochs, validation=None,
              record_every_epoch=False):
    x, y = arrays(records, embeddings)
    stats = scale_fit(x, y)
    refs = [r for r in records if r['pain_level'] == 0] if reference_mode else []
    assert not reference_mode or len(refs) >= 2
    architecture = [x.shape[1]*(3 if reference_mode else 1), 16, 1]
    tx, mask = inputs(records, embeddings, stats, refs)
    if validation:
        assert {r['file'] for r in validation}.isdisjoint({r['file'] for r in records})
        vx, vm = inputs(validation, embeddings, stats, refs)
        vy = arrays(validation, embeddings)[1]
    torch.manual_seed(42)
    model = MLP(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(forward(model,tx,mask,stats['offset']),y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if record_every_epoch:
            model.eval()
            with torch.inference_mode():
                train_prediction = forward(model,tx,mask,stats['offset'])
                entry = dict(epoch=epoch, training_mae=float((train_prediction-y).abs().mean()),
                             training_mse=float((train_prediction-y).square().mean()))
                if validation:
                    pred = forward(model,vx,vm,stats['offset'])
                    entry.update(validation_mae=float((pred-vy).abs().mean()),
                                 validation_mse=float((pred-vy).square().mean()), predictions=pred.tolist())
            history.append(entry)
    checkpoint = dict(kind='reference_mlp' if reference_mode else 'audio_mlp',
                      architecture=architecture, state_dict=copy.deepcopy(model.state_dict()),
                      weight_decay=decay, epochs=epochs, learning_rate=1e-3, seed=42,
                      references=refs,
                      reference_embeddings=arrays(refs,embeddings)[0] if refs else torch.empty(0,x.shape[1]),
                      **stats)
    return checkpoint, history


def predict(checkpoint, records, embeddings):
    model = MLP(checkpoint['architecture']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    merged = dict(embeddings)
    for record, embedding in zip(checkpoint['references'], checkpoint['reference_embeddings']):
        merged[record['file']] = embedding
    x, mask = inputs(records, merged, checkpoint, checkpoint['references'])
    with torch.inference_mode():
        return forward(model,x,mask,checkpoint['offset'])


def run_variant(name, calibration, evaluation, embeddings, reference_mode, decays, output, provenance):
    out = output/name
    out.mkdir()
    plan = dict(calibration=calibration, evaluation=evaluation, reference_mode=reference_mode,
                reference_policy='Every pain-0 recording in the training fold; self-pairs excluded; average predictions across available references.',
                fold_policy='Train take 1 plus any extra calibration takes; validate take 2, then swap. Validation phrases are excluded from the reference pool.',
                architecture=[1536 if reference_mode else 512,16,1],
                weight_decay_candidates=decays, epoch_candidates=list(range(1,401)),
                seed=42, learning_rate=1e-3, activation='GELU', optimizer='AdamW', loss='MSE',
                encoder_training='Frozen selected TAME-trained ResNet-34.',
                selection='Minimum mean calibration-fold MAE; ties: stronger decay, then fewer epochs.',
                evaluation_curve_policy='Retrospective curve on the fixed evaluation phrases, after selecting and saving the checkpoint using calibration only.',
                evaluation_scope='Exploratory, reused within-recording split.', **provenance)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    candidates, all_curves = [], []
    for decay in decays:
        folds = []
        for take in (1,2):
            training = [r for r in calibration if r['take'] == take or r['take'] > 2]
            validation = [r for r in calibration if r['take'] == 3-take]
            _, history = train_run(training,embeddings,reference_mode,decay,400,validation,True)
            folds.append(dict(training_files=[r['file'] for r in training],
                              validation_files=[r['file'] for r in validation],
                              reference_files=[r['file'] for r in training if r['pain_level']==0] if reference_mode else [],
                              epochs=history))
        mean = [dict(epoch=i+1, **{key: float(np.mean([f['epochs'][i][key] for f in folds]))
                                  for key in ['training_mae','training_mse','validation_mae','validation_mse']}) for i in range(400)]
        all_curves.append(dict(weight_decay=decay,folds=folds,mean=mean))
        candidates.extend(dict(weight_decay=decay,epoch=r['epoch'],mae=r['validation_mae']) for r in mean)
        print(f'{name}, decay={decay}: best calibration CV MAE {min(r["validation_mae"] for r in mean):.4f}',flush=True)
    chosen = min(candidates,key=lambda r:(r['mae'],-r['weight_decay'],r['epoch']))
    (out/'calibration_curves.json').write_text(json.dumps(all_curves,indent=2))
    (out/'selection.json').write_text(json.dumps(dict(selected=chosen,candidates=candidates),indent=2))
    checkpoint, _ = train_run(calibration,embeddings,reference_mode,chosen['weight_decay'],chosen['epoch'])
    checkpoint.update(plan_sha256=sha(out/'plan.json'),encoder_checkpoint_sha256=provenance['encoder_checkpoint_sha256'])
    torch.save(checkpoint,out/'model.pt')
    prediction = predict(checkpoint,evaluation,embeddings)
    target = arrays(evaluation,embeddings)[1]
    rows = [dict(r,prediction=float(p),absolute_error=abs(float(p)-r['pain_level'])) for r,p in zip(evaluation,prediction)]
    report = dict(calibration_count=len(calibration),evaluation_count=len(evaluation),selected=chosen,
                  evaluation_mae=float((prediction-target).abs().mean()),
                  evaluation_rmse=float((prediction-target).square().mean().sqrt()),
                  recordings=rows,checkpoint_sha256=sha(out/'model.pt'))
    (out/'evaluation_results.json').write_text(json.dumps(report,indent=2))
    # Generate the requested evaluation curve after the selected result is fixed.
    _, curve = train_run(calibration,embeddings,reference_mode,chosen['weight_decay'],400,evaluation,True)
    np.testing.assert_allclose(curve[chosen['epoch']-1]['predictions'],prediction.tolist(),rtol=0,atol=1e-6)
    (out/'evaluation_curve.json').write_text(json.dumps(curve,indent=2))
    with (out/'evaluation_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'evaluation_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','training_mae','training_mse','validation_mae','validation_mse'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in curve)
    print(json.dumps(dict(variant=name,selected=chosen,evaluation_mae=report['evaluation_mae']),indent=2),flush=True)
    return report


def main():
    root = Path('runs/milan_pain/resnet_lamita_setup')
    if root.exists():
        raise FileExistsError(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = Path('runs/milan_pain/resnet_personal')
    plan = json.loads((baseline/'plan.json').read_text())
    baseline_settings=json.loads((baseline/'selection.json').read_text())['mlp']['selected']['mlp_16']
    cache=load(baseline/'embeddings.pt')
    for name,digest in cache['audio_sha256'].items():assert sha(Path('data/milan_pain')/name)==digest
    original_calibration, original_evaluation = plan['calibration'],plan['validation']
    extra = [r for r in original_evaluation if r['pain_level']==0]
    assert len(extra)==1
    matched_calibration=sorted(original_calibration+extra,key=lambda r:(r['pain_level'],r['take']))
    matched_evaluation=[r for r in original_evaluation if r['pain_level']!=0]
    assert len(matched_calibration)==17 and len(matched_evaluation)==10
    provenance=dict(encoder_checkpoint_sha256=plan['encoder_checkpoint_sha256'],
                    embeddings_sha256=sha(baseline/'embeddings.pt'),baseline_plan_sha256=sha(baseline/'plan.json'),
                    code_sha256=sha(__file__),torch_version=str(torch.__version__))
    root.mkdir()
    reports={}
    reports['original_audio']=run_variant('original_audio',original_calibration,original_evaluation,cache['embeddings'],False,[baseline_settings['weight_decay']],root,provenance)
    reports['matched_audio']=run_variant('matched_audio',matched_calibration,matched_evaluation,cache['embeddings'],False,[0.01,0.1,1.],root,provenance)
    reports['pain0_reference']=run_variant('pain0_reference',matched_calibration,matched_evaluation,cache['embeddings'],True,[0.01,0.1,1.],root,provenance)
    summary={name:{k:r[k] for k in ['calibration_count','evaluation_count','selected','evaluation_mae','evaluation_rmse']} for name,r in reports.items()}
    summary['matched_reference_mae_reduction']=reports['matched_audio']['evaluation_mae']-reports['pain0_reference']['evaluation_mae']
    (root/'comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
