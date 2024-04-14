

import pytest

from cbnn.model.modules.encoders import CNNVariationalEncoder

import torch


class TestCNNVariationalEncoder:
    @pytest.fixture
    def encoder(self):
        model = CNNVariationalEncoder(in_channels=1, image_dim=32, latent_dim=2)
        model.eval()
        return model

    def test_forward(self, encoder):
        x = torch.randn(1, 1, 32, 32)
        mu, log_var = encoder(x)
        assert mu.shape == (1, 2)
        assert log_var.shape == (1, 2)