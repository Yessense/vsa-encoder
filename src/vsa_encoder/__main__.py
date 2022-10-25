import os
import random
import sys
from pathlib import Path

sys.path.append("..")

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning import seed_everything
import numpy as np

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.vsa_vae import VSAVAE
from dataset.paired_dsprites import PairedDspritesDataset


def train(config) -> None:
    seed_everything(config.seed)

    # Logger
    wandb_logger = WandbLogger(project='sweeps_demo')

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    path_to_dataset = Path(config.path_to_dataset)
    images_path = path_to_dataset / 'dsprite_train.npz'
    train_path = path_to_dataset / 'paired_train.npz'
    test_path = path_to_dataset / 'paired_test.npz'

    train_dataset = PairedDspritesDataset(dsprites_path=images_path, paired_dsprites_path=train_path)
    test_dataset = PairedDspritesDataset(dsprites_path=images_path, paired_dsprites_path=test_path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=10, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=10, drop_last=True)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    # model
    dict_args = dict(config)
    dict_args['steps_per_epoch'] = len(train_loader)
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
                         max_steps=config.max_steps,
                         profiler=profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         check_val_every_n_epoch=5)

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
    print("hello")
    wandb.init(project="sweeps_demo", entity="yessense")

    config = wandb.config
    print(f'Starting a run with {config}')

    train(config)
