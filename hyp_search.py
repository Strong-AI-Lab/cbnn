
from argparse import Namespace

from run import parse_args, main

import torch
from ray import tune, train
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer



def hyp_search(config, **args):
    args = Namespace(**{**config, **args})
    main(args, callbacks=[RayTrainReportCallback()])

def build_config(args):
    config = {  
        # 'accelerator': 'gpu', 
        # 'devices': '1',
        # 'z_samples': tune.choice([4, 8, 16]),
        # 'w_samples': tune.choice([4, 8, 16]),
        'recon_weight': tune.uniform(0.1, 4),
        'kld_weight': tune.loguniform(1e-6, 1e-2),
        'context_kld_weight': tune.loguniform(1e-6, 1e-2),
        'w_kld_weight': tune.loguniform(1e-6, 1e-2),
        'ic_mi_weight': tune.loguniform(1e-6, 1e-2),
        'wc_mi_weight': tune.loguniform(1e-6, 1e-2),
        # 'learning_rate': tune.loguniform(1e-5, 1e-3),
        # 'weight_decay': tune.loguniform(1e-5, 1e-3),
        # 'latent_dim': tune.choice([32, 64, 128, 256, 512]),
        # 'encoder_hidden_dims': tune.choice([[32, 64, 128, 256, 512], [64, 128, 256, 512, 1024]]),
        # 'classifier_hidden_dim': tune.choice([32, 64, 128, 256, 512]),
        # 'classifier_nb_layers': tune.choice([1, 3, 6]),
    }
    for key in config:
        if key in args:
            del args[key]

    args['strategy'] = RayDDPStrategy()
    args['plugins'] = RayLightningEnvironment()
    args['enable_progress_bar'] = False
    
    return config, args


if __name__ == '__main__':
    args = parse_args()
    config, args = build_config(vars(args))
    
    trainable = tune.with_parameters(
        hyp_search,
        **args
    )
    ray_trainer = TorchTrainer(
        trainable,
        run_config=train.RunConfig(stop={"training_iteration": 20}),
        scaling_config=train.ScalingConfig(num_workers=3, use_gpu=True, resources_per_worker={"GPU": 1}),
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="train_loss",
            mode="min",
            num_samples=40,
        ),
    )

    results = tuner.fit()
    print(results.get_best_result(metric="val_Accuracy", mode="max").config)