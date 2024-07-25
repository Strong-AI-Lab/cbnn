
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
        'z_samples': tune.choice([1, 4]),
        'w_samples': tune.choice([1, 4]),
        'recon_weight': tune.loguniform(1e-2, 10),
        'kld_weight': tune.loguniform(1e-7, 1e-3),
        'context_kld_weight': tune.loguniform(1e-7, 1e-3),
        'w_kld_weight': tune.loguniform(1e-7, 1e-3),
        'context_inference_weight': tune.choice([0.0, 0.25, 0.4]),
        'context_split_mi_weight': tune.choice([0.0, 0.5, 1.0]),
        'split_recons_infer_latents': tune.choice([None, 0.5]),
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'recon_weight': tune.loguniform(1e-2, 10),
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