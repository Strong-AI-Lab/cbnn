

import pytest

from cbnn.data.datasets import ConceptARCDataModule
from cbnn.model.vae import CNNVAE


class TestConceptARCDataModule:
    @pytest.fixture
    def conceptarc_gen(self):
        data = ConceptARCDataModule(data_dir="data/CONCEPTARC", batch_size=2, num_workers=0, mode="generation")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    @pytest.fixture
    def conceptarc_infer(self):
        data = ConceptARCDataModule(data_dir="data/CONCEPTARC", batch_size=2, num_workers=0, mode="inference")
        data.prepare_data()
        data.setup("fit")
        data.setup("test")
        return data
    
    def test_conceptarc_setup_gen(self, conceptarc_gen):
        assert conceptarc_gen.train_dataloader() is not None
        assert conceptarc_gen.val_dataloader() is not None
        assert conceptarc_gen.test_dataloader() is not None

    def test_conceptarc_next_train_gen(self, conceptarc_gen):
        x_train, y_train = next(iter(conceptarc_gen.train_dataloader()))
        assert x_train.shape == (2, 10, 128, 128)
        assert y_train.shape == (2,)

    def test_conceptarc_next_val_gen(self, conceptarc_gen):
        x_val, y_val = next(iter(conceptarc_gen.val_dataloader()))
        assert x_val.shape == (2, 10, 128, 128)
        assert y_val.shape == (2,)

    def test_conceptarc_next_test_gen(self, conceptarc_gen):
        x_test, y_test = next(iter(conceptarc_gen.test_dataloader()))
        assert x_test.shape == (2, 10, 128, 128)
        assert y_test.shape == (2,)

    def test_conceptarc_setup_infer(self, conceptarc_infer):
        assert conceptarc_infer.train_dataloader() is not None
        assert conceptarc_infer.val_dataloader() is not None
        assert conceptarc_infer.test_dataloader() is not None

    def test_conceptarc_next_train_infer(self, conceptarc_infer):
        pass # TODO: Implement this test after fix of inference mode



    def test_forward_gen(self, conceptarc_gen):
        x = next(iter(conceptarc_gen.train_dataloader()))[0]
        vae = CNNVAE(in_channels=10, latent_dim=20, image_dim=128)
        vae.eval()
        x_recon, x_input, z_mean, z_log_var = vae(x)
        assert x_recon.shape == (2, 10, 128, 128)
        assert x_input.shape == (2, 10, 128, 128)
        assert z_mean.shape == (2, 20)
        assert z_log_var.shape == (2, 20)
        assert (x_input == x).all()