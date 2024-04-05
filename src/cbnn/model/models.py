
from .cbnn import CNN_CBNN
from .vae import CNNVAE


MODELS = {
    'cbnn': CNN_CBNN,
    'vae': CNNVAE
}

def add_model_specific_args(parent_parser):
    subparsers = parent_parser.add_subparsers(help='Model to train or load for inference.')
    for model_name, model in MODELS.items():
        subparser = subparsers.add_parser(model_name, help=f'{model_name} model')
        subparser = model.add_model_specific_args(subparser)
        subparser.set_defaults(model=model_name)
    return parent_parser


def get_model(model_name: str, **model_kwargs):
    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}')
    return MODELS[model_name](**model_kwargs)