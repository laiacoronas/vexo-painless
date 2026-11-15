"""Reproduce personal ResNet heads from the recorded split and source phrases."""
import argparse
import json
from pathlib import Path

import torch

from predict import PainModel, ROOT
from train import sha
from train_milan_resnet_reference import run_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person', choices=['milan', 'lamita'], required=True)
    parser.add_argument('--audio-root', type=Path, required=True,
                        help='Folder containing the recordings/ paths in the saved plan.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = PainModel()
    person = model.index['people'][args.person]
    plan = json.loads((ROOT / person['plan']).read_text())
    recordings = plan['calibration'] + plan['evaluation']
    for row in recordings:
        if sha(args.audio_root / row['file']) != person['audio_sha256'][row['file']]:
            raise ValueError(f"Source recording checksum mismatch: {row['file']}")
    embeddings = {row['file']: model.embed(args.audio_root / row['file'])[0] for row in recordings}
    args.output.mkdir(parents=True)
    torch.save({'embeddings': embeddings, 'audio_sha256': person['audio_sha256'],
                'encoder_checkpoint_sha256': model.index['encoder']['sha256']}, args.output / 'embeddings.pt')
    provenance = {
        'encoder_checkpoint_sha256': model.index['encoder']['sha256'],
        'embeddings_sha256': sha(args.output / 'embeddings.pt'),
        'source_plan_sha256': sha(ROOT / person['plan']),
        'code_sha256': {'train_person.py': sha(__file__),
                        'train_milan_resnet_reference.py': sha(ROOT / 'training/train_milan_resnet_reference.py')},
        'torch_version': str(torch.__version__),
    }
    torch.set_num_threads(1)
    reports = {}
    for name, reference_mode in [('matched_audio', False), ('pain0_reference', True)]:
        reports[name] = run_variant(name, plan['calibration'], plan['evaluation'], embeddings,
                                    reference_mode, plan['weight_decay_candidates'], args.output, provenance)
    summary = {name: {key: report[key] for key in
                      ['calibration_count', 'evaluation_count', 'selected', 'evaluation_mae', 'evaluation_rmse']}
               for name, report in reports.items()}
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
