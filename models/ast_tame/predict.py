"""Run the bundled AST encoder and TAME pain head on segmented WAV audio."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTModel

from train import Regressor, load

ROOT = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, head='general', device='cpu'):
        if head not in ('general', 'personal'):
            raise ValueError('head must be general or personal')
        self.device = torch.device(device)
        self.checkpoint = load(ROOT / 'heads' / f'{head}.pt')
        self.extractor = ASTFeatureExtractor.from_pretrained(ROOT / 'base', local_files_only=True)
        self.encoder = ASTModel.from_pretrained(ROOT / 'base', local_files_only=True).to(self.device).eval()
        self.encoder.requires_grad_(False)
        self.head = Regressor(self.checkpoint['width'], self.checkpoint['personal']).to(self.device).eval()
        self.head.load_state_dict(self.checkpoint['state_dict'])
        self.mean = self.checkpoint['mean'].to(self.device)
        self.std = self.checkpoint['std'].to(self.device)

    @torch.inference_mode()
    def embed(self, path):
        waveform, rate = sf.read(path, dtype='float32', always_2d=True)
        waveform = waveform.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f'Audio must contain finite samples: {path}')
        inputs = self.extractor(waveform, sampling_rate=16000, return_tensors='pt')
        embedding = self.encoder(**{k: v.to(self.device) for k, v in inputs.items()}).pooler_output
        if not torch.isfinite(embedding).all():
            raise ValueError('AST produced a non-finite embedding')
        return (embedding - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, query, references=()):
        references = list(references)
        if self.head.personal:
            if len(references) != 3:
                raise ValueError('The reference head requires three (audio path, pain rating) pairs')
            values = [float(score) for _, score in references]
            if any(not math.isfinite(score) or not 1 <= score <= 10 for score in values):
                raise ValueError('Reference ratings must be finite and within the TAME training scale, 1–10')
            refs = torch.stack([self.embed(path)[0] for path, _ in references])[None]
            scores = torch.tensor([values], device=self.device)
        else:
            if references:
                raise ValueError('Choose --head personal to use reference recordings')
            refs = scores = None
        result = self.head(self.embed(query), refs, scores).item()
        if not math.isfinite(result):
            raise ValueError('Head produced a non-finite prediction')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', type=Path, help='One segmented spoken phrase, as WAV')
    parser.add_argument('--head', choices=['general', 'personal'], default='general')
    parser.add_argument('--reference', nargs=2, action='append', default=[], metavar=('WAV', 'RATING'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.head, args.device)
    prediction = predictor.predict(args.query, args.reference)
    print(json.dumps({'head': args.head, 'prediction': prediction, 'training_scale': [1, 10],
                      'outside_training_scale': not 1 <= prediction <= 10}, indent=2))
