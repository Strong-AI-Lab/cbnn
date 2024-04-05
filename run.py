
import argparse
import yaml

from src.cbnn.data.datasets import DATASETS
from src.cbnn.model.models import get_model, add_model_specific_args

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test a model or load existing model for inference.')

    # Add arguments
    parser.add_argument('--data', type=str, default='MNIST', help=f'Dataset to use for training. Options: {", ".join(DATASETS.keys())}')
    parser.add_argument('--save', type=str, default=None, help='Path to saved model to load for inference. If None, train a new model from scratch.')
    parser.add_argument('--wandb_project', type=str, default=None, help='If specified, logs the run to wandb under the specified project.')
    parser.add_argument('--load_config', type=str, default=None, help='Path to a yaml config file to load arguments from. Arugments provided in the command line will override the config file.')
    parser.add_argument('--save_config', type=str, default=None, help='Path to save the current arguments to a config file.')

    # Add mutually exclusive group for training, testing, or both
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Only train the model.')
    group.add_argument('--test', action='store_true', help='Only test the model.')
    group.add_argument('--train_and_test', action='store_true', help='Train a new model and test it. Default option.')

    # Add PyTorch Lightning Trainer arguments an model specific arguments
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_model_specific_args(parser)

    args = parser.parse_args()

    # If load_config is specified, load the config file and update the arguments
    if args.load_config is not None:
        with open(args.load_config, 'r') as f:
            config = yaml.safe_load(f)
            args = argparse.Namespace(**config, **vars(args))

    # If save_config is specified, save the current arguments to a config file
    if args.save_config is not None:
        args_dict = vars(args)
        save_file = args_dict.pop('save_config') # do not keep the save_config argument in the saved config file to avoid overriding it when loading
        with open(save_file, 'w') as f:
            yaml.dump(args_dict, f)

    return args



def main():
    args = parse_args()

    # Load model
    if not hasattr(args, 'model'):
        print('Error: Model not specified. Please, provide a model to use. Use --help for more information.')
        exit()
    
    model = get_model(args.model, **vars(args))

    if args.save is not None:
        model.load_from_checkpoint(args.save)


    # Load data
    train_set, validation_set, test_set = DATASETS[args.data]()

    
    # Build trainer
    pl.seed_everything(42, workers=True)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=WandbLogger(name=f"{args.model}_train", project=args.wandb_project) if args.wandb_project else None
    )


    # Train and test model (default: train_and_test)
    is_train = not args.test
    is_test = not args.train

    if is_train:
        trainer.fit(model, train_set, validation_set)

    if is_test:
        trainer.test(model, test_set)




if __name__ == '__main__':
    main()