import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from resnet_experiment import (PredictionModel, backbone, embed, log_mel,
                               prefix, validate_manifest)


class ResNetExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.encoder = backbone(False).eval()

    def test_cached_prefix_matches_full_encoder(self):
        x = torch.randn(2, 1, 128, 300)
        with torch.no_grad():
            cached = prefix(self.encoder, x)
            torch.testing.assert_close(embed(self.encoder, cached), self.encoder(x))
            self.assertEqual(cached.shape, (2, 256, 8, 19))
            self.assertEqual(self.encoder(x).shape, (2, 512))
        self.assertEqual(sum(p.numel() for p in self.encoder.parameters()), 21278400)

    def test_finetuning_updates_final_block_and_preserves_batch_statistics(self):
        model = PredictionModel(self.encoder, torch.zeros(512), torch.ones(512), True)
        model.train()
        before_initial = self.encoder.layer4[0].conv1.weight.detach().clone()
        before_model = model.layer4[0].conv1.weight.detach().clone()
        before_bn = model.layer4[0].bn1.running_mean.clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pred = model(torch.randn(2, 256, 8, 19))
        nn.functional.huber_loss(pred, torch.tensor([2., 8.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before_model, model.layer4[0].conv1.weight))
        torch.testing.assert_close(before_initial, self.encoder.layer4[0].conv1.weight)
        torch.testing.assert_close(before_bn, model.layer4[0].bn1.running_mean)

    def test_heads_start_identically_and_paths_agree(self):
        torch.manual_seed(42)
        frozen = PredictionModel(self.encoder, torch.zeros(512), torch.ones(512), False).eval()
        torch.manual_seed(42)
        tuned = PredictionModel(self.encoder, torch.zeros(512), torch.ones(512), True).eval()
        x = torch.randn(2, 256, 8, 19)
        with torch.no_grad():
            torch.testing.assert_close(frozen(embed(self.encoder, x)), tuned(x))
        self.assertEqual(sum(p.numel() for p in frozen.parameters()), 73985)

    def test_preprocessing_resamples_pads_and_handles_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.wav'
            wave = .2 * np.sin(2 * np.pi * 220 * np.arange(24000) / 48000)
            sf.write(path, wave, 48000)
            mel, duration, frames = log_mel(path)
            self.assertEqual(mel.shape, (1, 128, 300))
            self.assertEqual(duration, .5)
            self.assertLess(frames, 300)
            self.assertTrue(torch.isfinite(mel).all())
            self.assertEqual(torch.count_nonzero(mel[:, :, frames:]).item(), 0)
            sf.write(path, np.zeros(16000), 16000)
            mel, _, _ = log_mel(path)
            self.assertEqual(torch.count_nonzero(mel).item(), 0)

    def test_split_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_manifest({'splits': {'train': ['a'], 'validation': ['a'], 'test': ['b']}})


if __name__ == '__main__':
    unittest.main()
