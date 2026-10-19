import unittest

import torch

from roma_pose.models.feature_injection import RoMaCorrespondenceFeatureInjection
from roma_pose.utils.geometry import challenge_normalize


class InjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = RoMaCorrespondenceFeatureInjection(feature_dim=4, geom_hidden=8, feat_hidden=8)
        self.src = torch.randn(1, 4, 4)
        self.tgt = torch.randn(1, 4, 4)
        self.pos = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])
        self.grid = torch.tensor([[[[-.5, -.5], [.5, -.5]], [[-.5, .5], [.5, .5]]]])
        self.cert = torch.ones(1, 1, 2, 2)

    def run_direction(self, grid=None, cert=None):
        return self.model._one_direction(self.src, self.tgt, self.pos,
                                         self.grid if grid is None else grid,
                                         self.cert if cert is None else cert, 2, 2)

    def test_identity_correspondence_samples_target_features(self):
        _, aux = self.run_direction()
        torch.testing.assert_close(aux["matched_tgt"], self.tgt)
        torch.testing.assert_close(aux["geom"][..., 4:], torch.zeros(1, 4, 2))

    def test_zero_start_preserves_features_and_delays_mlp_gradients(self):
        new, _ = self.run_direction()
        torch.testing.assert_close(new, self.src, rtol=0, atol=0)
        new.square().sum().backward()
        self.assertGreater(abs(self.model.alpha.grad.item()) + abs(self.model.beta.grad.item()), 0)
        for parameter in self.model.geom_mlp.parameters():
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_confidence_and_out_of_bounds_block_update(self):
        with torch.no_grad():
            self.model.alpha.fill_(1)
            self.model.beta.fill_(1)
        zero, _ = self.run_direction(cert=torch.zeros_like(self.cert))
        invalid, _ = self.run_direction(grid=torch.full_like(self.grid, 2))
        torch.testing.assert_close(zero, self.src, rtol=0, atol=0)
        torch.testing.assert_close(invalid, self.src, rtol=0, atol=0)

    def test_normalization_cancels_first_pose(self):
        first = torch.eye(4).unsqueeze(0)
        first[:, :3, 3] = torch.tensor([1., 2., 3.])
        current = first.clone()
        current[:, 0, 3] += 4
        expected = torch.eye(4).unsqueeze(0)
        expected[:, 0, 3] = 4
        torch.testing.assert_close(challenge_normalize(first, current), expected)
        torch.testing.assert_close(challenge_normalize(first, first), torch.eye(4).unsqueeze(0))


if __name__ == "__main__":
    unittest.main()
