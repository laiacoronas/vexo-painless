import unittest
import torch
from train import Regressor, participant_weights, metrics, device


class ModelTests(unittest.TestCase):
    def test_explicit_cpu_device(self):
        self.assertEqual(device('cpu'), torch.device('cpu'))

    def test_reference_permutation_and_anchor(self):
        model = Regressor(8, True).eval()
        query, refs = torch.randn(2, 8), torch.randn(2, 3, 8)
        scores = torch.tensor([[1., 4., 7.], [2., 3., 8.]])
        expected = model(query, refs, scores)
        order = [2, 0, 1]
        torch.testing.assert_close(expected, model(query, refs[:, order], scores[:, order]))
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        torch.testing.assert_close(model(query, refs, scores), scores.mean(1))

    def test_gradient_reaches_both_heads(self):
        for personal in [False, True]:
            model = Regressor(8, personal)
            pred = model(torch.randn(2, 8), torch.randn(2, 3, 8), torch.ones(2, 3))
            self.assertEqual(pred.shape, (2,))
            torch.nn.functional.huber_loss(pred, torch.tensor([2., 6.])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_equal_participant_influence(self):
        examples = [{'participant': p, 'query': {'score': 3., 'clip': str(i)}}
                    for i, p in enumerate(['a', 'a', 'b'])]
        torch.testing.assert_close(participant_weights(examples), torch.tensor([.5, .5, 1.]))
        result = metrics(examples, [3., 3., 7.])
        self.assertEqual(result['participant_macro_mae'], 2.)

    def test_ast_feature_and_embedding_interface(self):
        import numpy as np
        from transformers import ASTConfig, ASTFeatureExtractor, ASTModel
        extractor = ASTFeatureExtractor(num_mel_bins=32, max_length=32)
        inputs = extractor(np.zeros(16000, dtype=np.float32), sampling_rate=16000,
                           return_tensors='pt')
        config = ASTConfig(num_mel_bins=32, max_length=32, hidden_size=16,
                           num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        encoder = ASTModel(config).eval()
        with torch.inference_mode():
            embedding = encoder(**inputs).pooler_output
        self.assertEqual(embedding.shape, (1, 16))
        self.assertTrue(torch.isfinite(embedding).all())


if __name__ == '__main__':
    unittest.main()
