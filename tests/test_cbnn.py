
import pytest

from cbnn.model.cbnn import CNN_CBNN

import torch


class TestCNN_CBNN:
    @pytest.fixture
    def cbnn(self):
        model = CNN_CBNN(in_channels=3, latent_dim=10, image_dim=32, out_channels=4, classifier_hidden_dim=10, classifier_nb_layers=1, z_samples=3, w_samples=2, nb_input_images=2)
        model.eval()
        return model

    def test_forward(self, cbnn):
        x = torch.randn(1, 2, 3, 32, 32)
        x_recon, y_recon, w_samples, z_samples, context_z_samples, z_mean, z_log_var, context_z_mean, context_z_log_var = cbnn(x) 
        assert type(x_recon) == list and len(x_recon) == 3 and all([x_recon[i].shape == (1, 2, 3, 32, 32) for i in range(3)])
        assert y_recon.shape == (1, 4)
        assert type(w_samples) == list and len(w_samples) == 6
        assert type(z_samples) == list and len(z_samples) == 3
        assert type(context_z_samples) == list and len(context_z_samples) == 3
        assert z_mean.shape == (1, 20)
        assert z_log_var.shape == (1, 20)
        assert type(context_z_mean) == list and len(context_z_mean) == 1 and context_z_mean[0].shape == (1, 20)
        assert type(context_z_log_var) == list and len(context_z_log_var) == 1 and context_z_log_var[0].shape == (1, 20)

    def test_sample(self, cbnn):
        mean = torch.randn(1, 10)
        log_var = torch.randn(1, 10)
        z = cbnn.sample(mean, log_var)
        assert z.shape == (1, 10)

    def test_generate(self, cbnn):
        x = torch.randn(1, 3, 32, 32)
        x_recon = cbnn.generate(x)
        assert x_recon.shape == (1, 3, 32, 32)

    def test_decode(self, cbnn):
        z = torch.randn(1, 10)
        x_recon = cbnn.decode(z)
        assert x_recon.shape == (1, 3, 32, 32)

    def test_pre_load_context(self, cbnn):
        x = [torch.randn(1, 2, 3, 32, 32)]
        cbnn.pre_load_context(x)
        assert cbnn.loaded_context == True
        assert cbnn.x_context == x

    def test_clear_context(self, cbnn):
        x = [torch.randn(1, 2, 3, 32, 32)]
        cbnn.pre_load_context(x)
        cbnn.clear_context()
        assert cbnn.loaded_context == False
        assert cbnn.x_context == None

    def accuracy(self, cbnn):
        y = torch.tensor([0, 1, 2, 3])
        y_pred = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        acc = cbnn.accuracy(y, y_pred)
        assert acc == 1.0

    def test_loss_function(self, cbnn):
        x = torch.randn(4, 2, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        x_recon, y_recon, w_samples, z_samples, context_z_samples, z_mean, z_log_var, context_z_mean, context_z_log_var = cbnn(x)
        loss = cbnn.loss_function(x, x_recon, y, y_recon, w_samples, z_samples, context_z_samples, z_mean, z_log_var, context_z_mean, context_z_log_var)
        assert ["loss", "Inference_Loss", "Reconstruction_Loss", "Context_KLD", "KLD", "IC_MI", "Accuracy"] == list(loss.keys())
        assert loss["loss"] > 0
        assert loss["Inference_Loss"] > 0
        assert loss["Reconstruction_Loss"] > 0
        assert loss["Context_KLD"] < 0
        assert loss["KLD"] < 0
        assert loss["Accuracy"] >= 0.0
        assert loss["loss"] == loss["Inference_Loss"] + cbnn.recon_weight * loss["Reconstruction_Loss"] - cbnn.context_kld_weight * loss["Context_KLD"] - cbnn.kld_weight * loss["KLD"] + cbnn.ic_mi_weight * loss["IC_MI"]

    