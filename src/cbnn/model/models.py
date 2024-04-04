
from .cbnn import CBNN
from .cnn_vi import CNNVAE


MODELS = {
    'cbnn': CBNN,
    'vae': CNNVAE
}


def get_model(model_name: str, **model_kwargs):
    return MODELS[model_name](**model_kwargs)