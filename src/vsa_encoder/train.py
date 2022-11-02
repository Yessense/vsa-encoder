import os
import random
import sys

import wandb

sys.path.append("..")
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import numpy as np

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from model.vsa_vae import VSAVAE
from dataset.paired_dsprites import PairedDspritesDataset


# from dataset import

def train(config):
    if isinstance(config, Namespace):
        dict_args = vars(config)
    else:
        dict_args = dict(config)

    # ------------------------------------------------------------
    # Random
    # ------------------------------------------------------------
    seed_everything(config.seed)

    # ------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------
    wandb_logger = WandbLogger(project=config.mode + '_vsa',
                               name=f'{config.mode} -l {config.latent_dim} '
                                    f'-s {config.seed} -kl {config.kld_coef}'
                                    f' -bs {config.batch_size}'
                                    f'vsa',
                               log_model=True)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    images_path = config.path_to_dataset / 'dsprite_train.npz'
    train_path = config.path_to_dataset / 'paired_train.npz'
    test_path = config.path_to_dataset / 'paired_test.npz'

    train_dataset = PairedDspritesDataset(dsprites_path=images_path, paired_dsprites_path=train_path)
    test_dataset = PairedDspritesDataset(dsprites_path=images_path, paired_dsprites_path=test_path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=10, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=10, drop_last=True)

    dict_args['steps_per_epoch'] = len(train_loader)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    autoencoder = VSAVAE(**dict_args)

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------

    monitor = 'Validation/Total'

    # checkpoints
    save_top_k = 1
    top_metric_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)
    every_epoch_callback = ModelCheckpoint(every_n_epochs=10)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        top_metric_callback,
        every_epoch_callback,
        lr_monitor,
    ]

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------

    # trainer parameters
    profiler = 'simple'  # 'simple'/'advanced'/None
    devices = [int(config.devices)]

    # trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=devices,
                         max_epochs=config.max_epochs,
                         profiler=profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         check_val_every_n_epoch=5,
                         gradient_clip_val=5.0)

    if 'ckpt_path' not in dict_args:
        dict_args['ckpt_path'] = None

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    # if args.test:
    #     trainer.test(autoencoder.load_from_checkpoint(checkpoint_path=dict_args['ckpt_path']))
    # else:
    trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=test_loader,
                ckpt_path=dict_args['ckpt_path'])


if __name__ == '__main__':
    # ------------------------------------------------------------
    # Parse args
    # ------------------------------------------------------------

    parser = ArgumentParser()

    # add PROGRAM level args
    program_parser = parser.add_argument_group('program')

    # logger parameters
    program_parser.add_argument("--log_model", default=True)
    program_parser.add_argument("--logger_dir", type=str, default=None)

    # dataset parameters
    program_parser.add_argument("--mode", type=str, choices=['dsprites', 'clevr'], default='dsprites')
    program_parser.add_argument("--path_to_dataset", type=Path, default=Path(__file__).absolute().parent / "data",
                                help="Path to the dataset directory")

    # Experiment parameters
    program_parser.add_argument("--batch_size", type=int, default=4)
    program_parser.add_argument("--test", type=bool, default=False)
    program_parser.add_argument("--seed", type=int, default=42)

    # Add model specific args
    parser = VSAVAE.add_model_specific_args(parent_parser=parser)

    # Add all the available trainer options to argparse#
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse input
    config = parser.parse_args()

    print(f'Starting a run with {config}')

    train(config)
