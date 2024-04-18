
import argparse
import yaml

from src.cbnn.data.datasets import DATASETS, BaseDataModule, get_dataset
from src.cbnn.model.models import MODELS, get_model, add_model_specific_args

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description=f'Train and test a model or load existing model for inference. Available models: {", ".join(MODELS.keys())}')

    # Add arguments
    parser.add_argument('--data', type=str, default='MNIST', help=f'Dataset to use for training. Options: {", ".join(DATASETS.keys())}')
    parser.add_argument('--save', type=str, default=None, help='Path to saved model to load for inference. If None, train a new model from scratch.')
    parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')

    group = parser.add_argument_group()
    group.add_argument('--load_config', type=str, default=None, help='Path to a yaml config file to load arguments from. No arguments should be provided in the command line.')
    group.add_argument('--save_config', type=str, default=None, help='Path to save the current arguments to a config file.')

    # Add mutually exclusive group for training, testing, or both
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Only train the model.')
    group.add_argument('--test', action='store_true', help='Only test the model.')
    group.add_argument('--train_and_test', action='store_true', help='Train a new model and test it. Default option.')

    # Add PyTorch Lightning Trainer arguments an model specific arguments
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_model_specific_args(parser)
    parser = BaseDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    # If load_config is specified, load the config file and ignore all other arguments
    if args.load_config is not None:
        with open(args.load_config, 'r') as f:
            config = yaml.safe_load(f)
            args = argparse.Namespace(**config)

    # If save_config is specified, save the current arguments to a config file
    elif args.save_config is not None:
        args_dict = vars(args)
        save_file = args_dict.pop('save_config') # do not keep the save_config argument in the saved config file to avoid overriding it when loading
        with open(save_file, 'w') as f:
            yaml.dump(args_dict, f)

    return args



def main(args, callbacks=None):
    # Load model
    if not hasattr(args, 'model'):
        print('Error: Model not specified. Please, provide a model to use. Use --help for more information.')
        exit()
    
    model = get_model(args.model, **vars(args))

    if args.save is not None:
        model.load_from_checkpoint(args.save)


    # Load data (train, val, and test sets are loaded upon calling fit or test methods in the trainer)
    data = get_dataset(args.data, **vars(args))

    
    # Build trainer
    pl.seed_everything(42, workers=True)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=WandbLogger(name=f"{args.model}_train", project=args.wandb_project) if args.wandb_project else True,
        callbacks = callbacks
    )


    # Train and test model (default: train_and_test)
    is_train = not args.test
    is_test = not args.train

    if is_train:
        trainer.fit(model, data)

    if is_test:
        trainer.test(model, data)




if __name__ == '__main__':
    args = parse_args()
    main(args)