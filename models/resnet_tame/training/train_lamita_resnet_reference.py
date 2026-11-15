"""Apply Milan's frozen-ResNet pain-0-reference protocol to Lamita."""
import json
from pathlib import Path

import torch

from resnet_experiment import backbone, log_mel, PredictionModel
from train import load, sha
from train_milan_resnet_reference import run_variant


def main():
    output = Path('runs/lamita_pain/resnet_reference')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    labels_path = Path('data/lamita_pain/labels.json')
    labels = json.loads(labels_path.read_text())
    old = json.loads(Path('runs/lamita_pain/predictions.json').read_text())
    old_rows = {r['file']:r for r in old['recordings']}
    encoder_path = Path('runs/resnet_comparison/pain_model.pt')
    milan_plan = json.loads(Path('runs/milan_pain/resnet_lamita_setup/pain0_reference/plan.json').read_text())
    assert sha(encoder_path) == old['checkpoint_sha256'] == milan_plan['encoder_checkpoint_sha256']
    grouped = {}
    for r in labels['recordings']:
        grouped.setdefault(r['pain_level'],[]).append(r)
    assert set(grouped) == {0,4,6,10} and len(grouped[0]) == 3
    calibration, evaluation = [], []
    for level,records in sorted(grouped.items()):
        records.sort(key=lambda r:r['source_start_seconds'])
        if level == 0:
            calibration.extend(records)
        else:
            calibration.extend(records[:2]);evaluation.extend(records[2:])
    assert len(calibration)==9 and len(evaluation)==6
    assert {r['file'] for r in calibration}.isdisjoint({r['file'] for r in evaluation})
    bundle = load(encoder_path)
    encoder = backbone(False)
    encoder.load_state_dict(bundle['encoder'])
    encoder.eval().requires_grad_(False)
    original_head = PredictionModel(encoder,bundle['mean'],bundle['std'],False).eval()
    original_head.head.load_state_dict(bundle['head'])
    output.mkdir()
    embeddings, hashes, audit = {}, {}, []
    with torch.inference_mode():
        for i,row in enumerate(labels['recordings'],1):
            path = labels_path.parent/row['file']
            hashes[row['file']] = sha(path)
            assert hashes[row['file']] == old_rows[row['file']]['audio_sha256']
            mel,duration,frames=log_mel(path)
            embedding=encoder(mel[None])[0]
            assert embedding.shape==(512,) and torch.isfinite(embedding).all()
            embeddings[row['file']]=embedding.clone()
            prediction=float(original_head(embedding[None]).item())
            difference=abs(prediction-old_rows[row['file']]['predicted_pain'])
            assert difference<1e-4,difference
            audit.append(dict(file=row['file'],duration_seconds=duration,mel_frames_available=frames,
                              original_prediction_difference=difference))
            print(f'Embedded {i}/15: {path.name}',flush=True)
    assert all(torch.equal(v,bundle['encoder'][k]) for k,v in encoder.state_dict().items())
    torch.save(dict(embeddings=embeddings,audio_sha256=hashes,encoder_checkpoint_sha256=sha(encoder_path)),output/'embeddings.pt')
    (output/'extraction_audit.json').write_text(json.dumps(dict(recordings=audit,encoder_weights_unchanged=True),indent=2))
    provenance=dict(encoder_checkpoint_sha256=sha(encoder_path),embeddings_sha256=sha(output/'embeddings.pt'),
                    labels_sha256=sha(labels_path),milan_protocol_plan_sha256=sha('runs/milan_pain/resnet_lamita_setup/pain0_reference/plan.json'),
                    code_sha256={name:sha(name) for name in [__file__,'train_milan_resnet_reference.py','resnet_experiment.py','train_personal.py','train_personal_mlp_sweep.py']},
                    torch_version=str(torch.__version__))
    torch.set_num_threads(1)
    reports={}
    for name,reference_mode in [('matched_audio',False),('pain0_reference',True)]:
        reports[name]=run_variant(name,calibration,evaluation,embeddings,reference_mode,[0.01,0.1,1.],output,provenance)
    comparison={name:{k:r[k] for k in ['calibration_count','evaluation_count','selected','evaluation_mae','evaluation_rmse']} for name,r in reports.items()}
    comparison['matched_reference_mae_reduction']=reports['matched_audio']['evaluation_mae']-reports['pain0_reference']['evaluation_mae']
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    main()
