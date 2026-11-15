"""Run the bundled ResNet and a general or person-specific pain head."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training'))
from resnet_experiment import PredictionModel, backbone, log_mel
from train import load, sha
from train_milan_resnet_reference import predict as predict_personal


class PainModel:
    def __init__(self, head='general'):
        self.index = json.loads((ROOT / 'model_index.json').read_text())
        self.name = head
        if head != 'general' and head not in self.index['heads']:
            raise ValueError(f'Unknown head: {head}')
        path = ROOT / self.index['encoder']['file']
        if sha(path) != self.index['encoder']['sha256']:
            raise ValueError('Encoder checksum mismatch. Retrieve the model with git lfs pull.')
        bundle = load(path)
        self.encoder = backbone(False)
        self.encoder.load_state_dict(bundle['encoder'])
        self.encoder.eval().requires_grad_(False)
        if head == 'general':
            self.head = PredictionModel(self.encoder, bundle['mean'], bundle['std'], False).eval()
            self.head.head.load_state_dict(bundle['head'])
        else:
            entry = self.index['heads'][head]
            head_path = ROOT / entry['file']
            if sha(head_path) != entry['sha256']:
                raise ValueError('Head checksum mismatch.')
            self.head = load(head_path)
            if self.head['encoder_checkpoint_sha256'] != self.index['encoder']['sha256']:
                raise ValueError('Head and encoder provenance differ.')

    def embed(self, audio):
        mel, duration, frames = log_mel(audio)
        with torch.inference_mode():
            embedding = self.encoder(mel.unsqueeze(0))[0]
        if not torch.isfinite(embedding).all():
            raise ValueError('Audio produced a non-finite embedding.')
        return embedding, duration, frames

    def predict_embedding(self, embedding, audio_sha256):
        with torch.inference_mode():
            if self.name == 'general':
                return float(self.head(embedding.unsqueeze(0)).item())
            # A reference reused as a query receives the original self-pair mask.
            key = 'query:' + audio_sha256
            for filename, digest in self.index['heads'][self.name]['reference_audio_sha256'].items():
                if digest == audio_sha256:
                    key = filename
                    break
            return float(predict_personal(self.head, [{'file': key, 'pain_level': 0}],
                                          {key: embedding}).item())

    def predict(self, audio):
        audio = Path(audio)
        embedding, duration, frames = self.embed(audio)
        value = self.predict_embedding(embedding, sha(audio))
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError('Prediction must be finite.')
        return {
            'head': self.name, 'predicted_pain': value,
            'outside_0_10': not 0 <= value <= 10,
            'training_label_range': [1, 10] if self.name == 'general' else [0, 10],
            'audio_duration_seconds': duration, 'mel_frames_available': frames,
            'mel_frames_used': min(frames, 300),
            'reference_count': 0 if self.name == 'general' else len(self.head['references']),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path, help='One segmented speech phrase.')
    parser.add_argument('--head', choices=['general', 'milan', 'lamita', 'milan-audio', 'lamita-audio'],
                        default='general')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(PainModel(args.head).predict(args.audio), indent=2, allow_nan=False))
