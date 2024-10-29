import argparse
import numpy as np
import pickle
import time

import torch


import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from callbacks.sklearn_metrics_test import TestMetricsSklearn
from feeders.probing_datamodule import ProbingDataModule
from model.probing import NonLinearProbe, LinearProbe, BinaryProbingModule


MODELS = {
    "linear": LinearProbe,
    "nonlinear": NonLinearProbe
}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Path to pickled data")
    parser.add_argument(
        "--probing_property",
        type=str,
        choices=['handedness', 'position', 'shape', 'movement', 'rotation'],
        help="Properties of gestures to probe representations for")
    
    parser.add_argument(
        "--probing_model",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear"],
        help="Probing model"
    )    
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        '--ignore_class_weights',
        action="store_true",
        default=False,
        help='Flag for ignoring positive class weight in BCE. By default: use class weights based on proportions of samples.',
    )
    
    parser.add_argument(
        '--wandb_entity',
        default='sensor_har',
        choices=["sensor_har", "none"],
        help='Name of wandb logger. Set to none not to use wandb and use tensorboard only.',
        type=str
    )
    return parser


def get_data_from_pickle(pickle_path):
    data = pickle.load(open(pickle_path, 'rb'))
    return data


def run_probing(args):
    data_df = get_data_from_pickle(args.data_path)
    concatenated_features = np.array([np.array(l) for l in data_df["concatenated_features"]])
    probing_labels = data_df[args.probing_property]
    print(
        f"Gesture property: {args.probing_property}. \n"
        f"Class distribution: \n{probing_labels.value_counts()}"
    )
    probing_labels = np.array(probing_labels)
    probing_labels[probing_labels == 9] = 0

    pos_class_weight = 1
    if not args.ignore_class_weights:
        pos_class_weight = (probing_labels == 0).sum() / (probing_labels == 1).sum() 

    model = BinaryProbingModule(
        MODELS[args.probing_model](concatenated_features.shape[1]), 
        pos_weight=pos_class_weight
    )

    datamodule = ProbingDataModule(concatenated_features, probing_labels) 

    loggers = []
    logger_name = "LinearProbe_"
    wandb_logger = WandbLogger(
        config=vars(args),
        entity=args.wandb_entity,
        project="CABB_probing",
        name=logger_name + str(time.time()), # temporary solution for unique experiment names in wandb
        id=logger_name + str(time.time())
    )
    loggers.append(wandb_logger)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=50,
        logger=loggers,
        callbacks = [
            TestMetricsSklearn()
        ]
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_probing(args)