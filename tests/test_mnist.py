

import pytest

from cbnn.data.datasets import MNISTDataModule
from cbnn.model.vae import CNNVAE


class TestMNISTDataModule:
    @pytest.fixture
    def mnist(self):
        data = MNISTDataModule(data_dir="data/MNIST", batch_size=2, num_workers=0)
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    def test_mnist_setup(self, mnist):
        assert mnist.train_dataloader() is not None
        assert mnist.val_dataloader() is not None
        assert mnist.test_dataloader() is not None

    def test_mnist_next_train(self, mnist):
        x_train, y_train = next(iter(mnist.train_dataloader()))
        assert x_train.shape == (2, 1, 32, 32)
        assert y_train.shape == (2,)

    def test_mnist_next_val(self, mnist):
        x_val, y_val = next(iter(mnist.val_dataloader()))
        assert x_val.shape == (2, 1, 32, 32)
        assert y_val.shape == (2,)

    def test_mnist_next_test(self, mnist):
        x_test, y_test = next(iter(mnist.test_dataloader()))
        assert x_test.shape == (2, 1, 32, 32)
        assert y_test.shape == (2,)

    def test_forward(self, mnist):
        x = next(iter(mnist.train_dataloader()))[0]
        vae = CNNVAE(in_channels=1, latent_dim=10, image_dim=32)
        vae.eval()
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (2, 1, 32, 32)
        assert x_input.shape == (2, 1, 32, 32)
        assert z_mean.shape == (2, 10)
        assert z_log_var.shape == (2, 10)
        assert (x_input == x).all()