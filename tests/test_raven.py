
import pytest

from cbnn.data.datasets import RAVENDataModule
from cbnn.model.vae import CNNVAE


class TestRAVENDataModule:
    @pytest.fixture
    def raven_gen(self):
        data = RAVENDataModule(data_dir="data/RAVEN/RAVEN-10000/", batch_size=2, num_workers=0, split="IID", mode="generation")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    @pytest.fixture
    def raven_infer(self):
        data = RAVENDataModule(data_dir="data/RAVEN/RAVEN-10000/", batch_size=2, num_workers=0, split="IID", mode="inference")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    def test_raven_setup_gen(self, raven_gen):
        assert raven_gen.train_dataloader() is not None
        assert raven_gen.val_dataloader() is not None
        assert raven_gen.test_dataloader() is not None

    def test_raven_next_train_gen(self, raven_gen):
        x_train, y_train = next(iter(raven_gen.train_dataloader()))
        assert x_train.shape == (2, 1, 128, 128)
        assert y_train.shape == (2,)

    def test_raven_next_val_gen(self, raven_gen):
        x_val, y_val = next(iter(raven_gen.val_dataloader()))
        assert x_val.shape == (2, 1, 128, 128)
        assert y_val.shape == (2,)

    def test_raven_next_test_gen(self, raven_gen):
        x_test, y_test = next(iter(raven_gen.test_dataloader()))
        assert x_test.shape == (2, 1, 128, 128)
        assert y_test.shape == (2,)

    def test_raven_setup_infer(self, raven_infer):
        assert raven_infer.train_dataloader() is not None
        assert raven_infer.val_dataloader() is not None
        assert raven_infer.test_dataloader() is not None

    def test_raven_next_train_infer(self, raven_infer):
        x_train, y_train = next(iter(raven_infer.train_dataloader()))
        assert x_train.shape == (2, 16, 1, 128, 128)
        assert y_train.shape == (2,)

    def test_raven_next_val_infer(self, raven_infer):
        x_val, y_val = next(iter(raven_infer.val_dataloader()))
        assert x_val.shape == (2, 16, 1, 128, 128)
        assert y_val.shape == (2,)

    def test_raven_next_test_infer(self, raven_infer):
        x_test, y_test = next(iter(raven_infer.test_dataloader()))
        assert x_test.shape == (2, 16, 1, 128, 128)
        assert y_test.shape == (2,)

    def test_forward_gen(self, raven_gen):
        x = next(iter(raven_gen.train_dataloader()))[0]
        vae = CNNVAE(in_channels=1, latent_dim=10, image_dim=128)
        vae.eval()
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (2, 1, 128, 128)
        assert x_input.shape == (2, 1, 128, 128)
        assert z_mean.shape == (2, 10)
        assert z_log_var.shape == (2, 10)
        assert (x_input == x).all()