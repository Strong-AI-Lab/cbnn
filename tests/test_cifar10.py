

import pytest

from cbnn.data.datasets import CIFAR10DataModule
from cbnn.model.vae import CNNVAE


class TestCIFAR10DataModule:
    @pytest.fixture
    def cifar10(self):
        data = CIFAR10DataModule(data_dir="data/CIFAR10", batch_size=2, num_workers=0)
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    def test_cifar10_setup(self, cifar10):
        assert cifar10.train_dataloader() is not None
        assert cifar10.val_dataloader() is not None
        assert cifar10.test_dataloader() is not None

    def test_cifar10_next_train(self, cifar10):
        x_train, y_train = next(iter(cifar10.train_dataloader()))
        assert x_train.shape == (2, 3, 32, 32)
        assert y_train.shape == (2,)

    def test_cifar10_next_val(self, cifar10):
        x_val, y_val = next(iter(cifar10.val_dataloader()))
        assert x_val.shape == (2, 3, 32, 32)
        assert y_val.shape == (2,)

    def test_cifar10_next_test(self, cifar10):
        x_test, y_test = next(iter(cifar10.test_dataloader()))
        assert x_test.shape == (2, 3, 32, 32)
        assert y_test.shape == (2,)

    def test_forward(self, cifar10):
        x = next(iter(cifar10.train_dataloader()))[0]
        vae = CNNVAE(in_channels=3, latent_dim=10, image_dim=32)
        vae.eval()
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (2, 3, 32, 32)
        assert x_input.shape == (2, 3, 32, 32)
        assert z_mean.shape == (2, 10)
        assert z_log_var.shape == (2, 10)
        assert (x_input == x).all()