
import pytest

from cbnn.data.datasets import ACREDataModule
from cbnn.model.vae import CNNVAE


class TestACREDataModule:
    @pytest.fixture
    def acre_gen(self):
        data = ACREDataModule(data_dir="data/ACRE", split="IID", batch_size=2, num_workers=0, mode="generation")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    @pytest.fixture
    def acr_infer(self):
        data = ACREDataModule(data_dir="data/ACRE", split="IID", batch_size=2, num_workers=0, mode="inference")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    def test_acre_setup_gen(self, acre_gen):
        assert acre_gen.train_dataloader() is not None
        assert acre_gen.val_dataloader() is not None
        assert acre_gen.test_dataloader() is not None

    def test_acre_next_train_gen(self, acre_gen):
        x_train, y_train = next(iter(acre_gen.train_dataloader()))
        assert x_train.shape == (2, 3, 256, 256)
        assert y_train.shape == (2,)

    def test_acre_next_val_gen(self, acre_gen):
        x_val, y_val = next(iter(acre_gen.val_dataloader()))
        assert x_val.shape == (2, 3, 256, 256)
        assert y_val.shape == (2,)

    def test_acre_next_test_gen(self, acre_gen):
        x_test, y_test = next(iter(acre_gen.test_dataloader()))
        assert x_test.shape == (2, 3, 256, 256)
        assert y_test.shape == (2,)

    def test_acre_setup_infer(self, acr_infer):
        assert acr_infer.train_dataloader() is not None
        assert acr_infer.val_dataloader() is not None
        assert acr_infer.test_dataloader() is not None

    def test_acre_next_train_infer(self, acr_infer):
        x_train, y_train = next(iter(acr_infer.train_dataloader()))
        assert x_train.shape == (2, 7, 3, 256, 256)
        assert y_train.shape == (2,)

    def test_acre_next_val_infer(self, acr_infer):
        x_val, y_val = next(iter(acr_infer.val_dataloader()))
        assert x_val.shape == (2, 7, 3, 256, 256)
        assert y_val.shape == (2,)

    def test_acre_next_test_infer(self, acr_infer):
        x_test, y_test = next(iter(acr_infer.test_dataloader()))
        assert x_test.shape == (2, 7, 3, 256, 256)
        assert y_test.shape == (2,)
    
    def test_forward_gen(self, acre_gen):
        x = next(iter(acre_gen.train_dataloader()))[0]
        vae = CNNVAE(in_channels=3, latent_dim=10, image_dim=256)
        vae.eval()
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (2, 3, 256, 256)
        assert x_input.shape == (2, 3, 256, 256)
        assert z_mean.shape == (2, 10)
        assert z_log_var.shape == (2, 10)
        assert (x_input == x).all()





    