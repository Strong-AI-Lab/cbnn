
from .cbnn import CNN_CBNN, MCQA_CNN_CBNN, ResNet_CBNN, MCQA_ResNet_CBNN, MIPred_ResNet_CBNN
from .vae import CNNVAE, CNNVAEClassifier
from .resnet import ResNet18, MCQAResNet18, MIPredResNet18
from .vit import VITB16, VITB32, VITL16, VITL32, VITH14, MIPredVITB16, MIPredVITB32, MIPredVITL16, MIPredVITL32, MIPredVITH14


MODELS = {
    'cbnn': CNN_CBNN,
    'mcqa_cbnn': MCQA_CNN_CBNN,
    'cbresnet': ResNet_CBNN,
    'mcqa_cbresnet': MCQA_ResNet_CBNN,
    'mipred_cbresnet': MIPred_ResNet_CBNN,
    'vae': CNNVAE,
    'vae_classifier': CNNVAEClassifier,
    'resnet18': ResNet18,
    'mcqa_resnet18': MCQAResNet18,
    'mipred_resnet18': MIPredResNet18,
    'vitb16': VITB16,
    'mipred_vitb16': MIPredVITB16,
    'vitb32': VITB32,
    'mipred_vitb32': MIPredVITB32,
    'vitl16': VITL16,
    'mipred_vitl16': MIPredVITL16,
    'vitl32': VITL32,
    'mipred_vitl32': MIPredVITL32,
    'vith14': VITH14,
    'mipred_vith14': MIPredVITH14,
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