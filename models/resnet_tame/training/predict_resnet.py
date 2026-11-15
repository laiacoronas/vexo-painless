"""Run the exported research model on an audio recording."""
import argparse
import json
from pathlib import Path

import torch

from resnet_experiment import PredictionModel, backbone, log_mel
from train import device, load


def predict(path, checkpoint, target):
    bundle = load(checkpoint)
    encoder = backbone(False)
    encoder.load_state_dict(bundle['encoder'])
    encoder.to(target).eval()
    head = PredictionModel(encoder, bundle['mean'], bundle['std'], False)
    head.head.load_state_dict(bundle['head'])
    head.to(target).eval()
    mel, duration, frames = log_mel(path)
    with torch.inference_mode():
        value = float(head(encoder(mel.unsqueeze(0).to(target))).cpu().item())
    return {'predicted_pain': value, 'training_label_range': [1, 10],
            'outside_training_label_range': not 1 <= value <= 10,
            'selected_variant': bundle['selected_variant'],
            'audio_duration_seconds': duration, 'mel_frames_used': min(frames, 300),
            'mel_frames_available': frames}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/resnet_comparison/pain_model.pt'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict(args.audio, args.checkpoint, device(args.device)), indent=2))
