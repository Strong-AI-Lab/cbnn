
import pytest

from cbnn.model.vae import CNNVAE

import torch


class TestCNNVAE:
    @pytest.fixture
    def vae(self):
        model = CNNVAE(in_channels=3, latent_dim=10, image_dim=32)
        model.eval()
        return model

    def test_forward(self, vae):
        x = torch.randn(1, 3, 32, 32)
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (1, 3, 32, 32)
        assert x_input.shape == (1, 3, 32, 32)
        assert z_mean.shape == (1, 10)
        assert z_log_var.shape == (1, 10)
        assert (x_input == x).all()

    def test_sample(self, vae):
        mean = torch.randn(1, 10)
        log_var = torch.randn(1, 10)
        z = vae.sample(mean, log_var)
        assert z.shape == (1, 10)

    def test_generate(self, vae):
        x = torch.randn(1, 3, 32, 32)
        x_recon = vae.generate(x)
        assert x_recon.shape == (1, 3, 32, 32)

    def test_encode(self, vae):
        x = torch.randn(1, 3, 32, 32)
        z_mean, z_log_var = vae.encode(x)
        assert z_mean.shape == (1, 10)
        assert z_log_var.shape == (1, 10)

    def test_decode(self, vae):
        z = torch.randn(1, 10)
        x_recon = vae.decode(z)
        assert x_recon.shape == (1, 3, 32, 32)

    def test_loss_function(self, vae):
        x = torch.randn(1, 3, 32, 32)
        x_recon, x_input, z_mean, z_log_var = vae(x)
        loss = vae.loss_function(x_input, x_recon, z_mean, z_log_var)
        assert ["loss", "Reconstruction_Loss", "KLD"] == list(loss.keys())
        assert loss["loss"] > 0
        assert loss["Reconstruction_Loss"] > 0
        assert loss["KLD"] < 0
        assert loss["loss"] == loss["Reconstruction_Loss"] - loss["KLD"] * vae.kld_weight



