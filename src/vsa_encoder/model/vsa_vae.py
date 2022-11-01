import math
from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import wandb
from torch.optim import lr_scheduler

from model.decoder import Decoder
from model.encoder import Encoder
import torch
from torch import nn
from utils import iou_pytorch
import torch.nn.functional as F


class VSAVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("VSA VAE")

        # dataset options
        parser.add_argument("--n_features", type=int, default=5)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(1, 64, 64))  # type: ignore
        parser.add_argument("--latent_dim", type=int, default=1024)

        # model options
        parser.add_argument("--lr", type=float, default=0.00025)
        parser.add_argument("--kld_coef", type=float, default=0.001)

        return parent_parser

    def __init__(self,
                 n_features: int = 5,
                 image_size: Tuple[int, int, int] = (1, 64, 64),
                 lr: float = 0.00030,
                 kld_coef: float = 0.001,
                 latent_dim: int = 1024,
                 **kwargs):
        super().__init__()
        # Experiment options
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_features = n_features

        # model parameters
        self.lr = lr
        self.kld_coef = kld_coef

        # Layers
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size, n_features=n_features)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size)

        # hd placeholders
        hd_placeholders = torch.randn(1, self.n_features, self.latent_dim)
        self.hd_placeholders = nn.Parameter(data=hd_placeholders)

        self.save_hyperparameters()

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def encode(self, x):
        """ Get features multiplied by placeholders

        x -> [batch_size, n_channels, height, width]
        out -> [batch_size, n_features, latent_dim]
        """
        mu, log_var = self.encoder(x)

        z = self.reparametrize(mu, log_var)
        z = z.reshape(-1, self.n_features, self.latent_dim)
        mask = self.hd_placeholders.data.expand(z.size())
        z = z * mask

        return z, mu, log_var

    def exchange(self, image_features, donor_features, exchange_labels):
        # Exchange
        exchange_labels = exchange_labels.expand(image_features.size())

        # Reconstruct image
        donor_features_exept_one = torch.where(exchange_labels, image_features, donor_features)
        donor_features_exept_one = torch.sum(donor_features_exept_one, dim=1)

        # Donor image
        image_features_exept_one = torch.where(exchange_labels, donor_features, image_features)
        image_features_exept_one = torch.sum(image_features_exept_one, dim=1)

        return donor_features_exept_one, image_features_exept_one

    def forward(self, image, donor, exchange_labels):
        image_features, image_mu, image_log_var = self.encode(image)
        donor_features, donor_mu, donor_log_var = self.encode(donor)

        donor_features_exept_one, image_features_exept_one = self.exchange(image_features, donor_features,
                                                                           exchange_labels)
        recon_like_image = self.decoder(donor_features_exept_one)
        recon_like_donor = self.decoder(image_features_exept_one)

        reconstructions = (recon_like_image, recon_like_donor)
        mus = (image_mu, donor_mu)
        log_vars = (image_log_var, donor_log_var)
        return reconstructions, mus, log_vars

    def _step(self, batch, batch_idx, mode='Train'):
        """ Base step"""

        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        elif mode == 'Test':
            log_images = lambda x: True
        else:
            raise ValueError

        image, donor, exchange_labels = batch
        reconstructions, mus, log_vars = self.forward(image, donor, exchange_labels)

        mus = sum(mus) * 2 ** -0.5
        log_vars = sum(mus) * 2 ** -0.5

        image_loss, donor_loss, kld_loss = self.loss_f((image, donor), reconstructions, mus, log_vars)

        total_loss = (image_loss + donor_loss) * 0.5 + self.kld_coef * kld_loss

        iou_image = iou_pytorch(reconstructions[0], image)
        iou_donor = iou_pytorch(reconstructions[1], donor)
        total_iou = (iou_image + iou_donor) / 2

        # ----------------------------------------
        # Logs
        # ----------------------------------------

        self.log(f"{mode}/Total", total_loss)
        self.log(f"{mode}/Reconstruct Image", image_loss)
        self.log(f"{mode}/Reconstruct Donor", donor_loss)
        self.log(f"{mode}/Mean Reconstruction", (image_loss + donor_loss) / 2)
        self.log(f"{mode}/KLD", kld_loss * self.kld_coef)
        self.log(f"{mode}/iou total", total_iou)
        self.log(f"{mode}/iou image", iou_image)
        self.log(f"{mode}/iou donor", iou_donor)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(donor[0], caption='Donor'),
                    wandb.Image(reconstructions[0][0], caption='Recon like Image'),
                    wandb.Image(reconstructions[1][0], caption='Recon like Donor'),
                ]})

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, mode='Validation')

    def loss_f(self, gt_images, reconstructions, mus, log_vars):
        reduction = 'mean'
        image_loss = F.mse_loss(gt_images[0], reconstructions[0], reduction=reduction)
        donor_loss = F.mse_loss(gt_images[1], reconstructions[1], reduction=reduction)

        kld_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        if reduction == 'mean':
            batch_size = mus.shape[0]
            kld_loss = kld_loss / batch_size

        return image_loss, donor_loss, kld_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                            epochs=self.hparams['max_epochs'],
                                            steps_per_epoch=self.hparams['steps_per_epoch'],
                                            pct_start=0.2)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }


if __name__ == '__main__':
    vsavae = VSAVAE()
    x = torch.randn(10, 1, 64, 64)
    exchanges = torch.randint(0, 2, (10, 5), dtype=bool)

    out = vsavae.forward(x, x, exchanges)
