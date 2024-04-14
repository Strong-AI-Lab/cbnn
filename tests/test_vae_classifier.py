
import pytest

from cbnn.model.vae import CNNVAEClassifier

import torch


class TestCNNVAEClassifier:
    @pytest.fixture
    def vae(self):
        model = CNNVAEClassifier(in_channels=3, latent_dim=10, image_dim=32, num_classes=4)
        model.eval()
        return model

    def test_forward(self, vae):
        x = torch.randn(1, 3, 32, 32)
        x_recon, y_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (1, 3, 32, 32)
        assert y_recon.shape == (1, 4)
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

    def test_classify(self, vae):
        z = torch.randn(1, 10)
        y = vae.classify(z)
        assert y.shape == (1, 4)

    def accuracy(self, vae):
        y = torch.tensor([0, 1, 2, 3])
        y_pred = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        acc = vae.accuracy(y, y_pred)
        assert acc == 1.0

    def test_loss_function(self, vae):
        x = torch.randn(1, 3, 32, 32)
        y = torch.tensor([0])
        x_recon, y_recon, x_input, z_mean, z_log_var = vae(x)
        loss = vae.loss_function(x_input, x_recon, y, y_recon, z_mean, z_log_var)
        assert ["loss", "Reconstruction_Loss", "KLD", "Inference_Loss", "Accuracy"] == list(loss.keys())
        assert loss["loss"] > 0
        assert loss["Reconstruction_Loss"] > 0
        assert loss["KLD"] < 0
        assert loss["Inference_Loss"] > 0
        assert loss["Accuracy"] >= 0.0
        assert loss["loss"] == loss["Reconstruction_Loss"] - loss["KLD"] * vae.kld_weight + loss["Inference_Loss"] * vae.inference_weight
