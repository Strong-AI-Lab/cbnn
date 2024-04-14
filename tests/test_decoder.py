

import pytest

from cbnn.model.modules.decoders import CNNVariationalDecoder

import torch

class TestCNNDecoder:
    @pytest.fixture
    def decoder(self):
        model = CNNVariationalDecoder(latent_dim=2, image_dim=32, in_channels=1)
        model.eval()
        return model
    
    def test_forward(self, decoder):
        z = torch.randn(1, 2)
        x = decoder(z)
        assert x.shape == (1, 1, 32, 32)