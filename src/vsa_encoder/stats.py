import itertools
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import sys

import matplotlib.pyplot as plt

from vsa import bind

sys.path.append("..")

import torch
import wandb
from pytorch_lightning import seed_everything

from model.vsa_vae import VSAVAE
from dataset.paired_dsprites import Dsprites
from torch.utils.data import DataLoader

SEED = 42
seed_everything(seed=SEED)


class Stats:
    def __init__(self, model_checkpoint_path: Path,
                 dataset_path: Path,
                 device: Optional[int] = None, **kwargs):

        # Model
        if torch.cuda.is_available() and device is not None:
            self.device = torch.device('cuda:' + str(device))
        else:
            self.device = torch.device('cpu')
        self.model: VSAVAE = VSAVAE.load_from_checkpoint(checkpoint_path=str(model_checkpoint_path)).to(self.device)
        self.model.eval()

        self.dataset = Dsprites(path=dataset_path)
        self.visualizer = Visualizer()

    def make_dataloader(self, batch_size: int = 4):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def restore_from_nth_features(self, n: int = 4, image_idx: int = 0):
        # get random image
        image, labels = self.dataset[image_idx]
        image = torch.from_numpy(image).to(self.device).float()
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        # get all possible combinations of n features
        used_features_list = list(itertools.combinations(range(self.model.n_features), n))
        for feature_comb in used_features_list:
            z, _, _ = self.model.encode(image)
            z = z[:, feature_comb, :]
            z = torch.sum(z, dim=1)
            decoded_image = self.model.decoder(z)
            title = 'Image decoded from features:\n' + \
                    ", ".join([f'{self.dataset.feature_names[feature_index]}' for feature_index in feature_comb])
            self.visualizer.plot_before_after(image[0], decoded_image[0], 'Image', 'Decoded Image', title)

            print(feature_comb)

    def simple_process_image(self,
                             feature_values,
                             multiply_by_placeholders: bool = False,
                             display: bool = True):
        # feature_names ('shape', 'scale', 'orientation', 'posX', 'posY')
        # features_count [3, 6, 40, 32, 32]
        image_idx: int = self.dataset._get_element_pos(feature_values)
        image, labels = self.dataset[image_idx]
        image = torch.from_numpy(image).to(self.device).float()
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        # labels = labels.to(self.device)

        features = self.model.get_features(image)

        if multiply_by_placeholders:
            placeholders = self.model.hd_placeholders.data

            if self.model.bind_mode == 'fourier':
                features = bind(features, placeholders)
            elif self.model.bind_mode == 'randn':
                features = features * placeholders
            else:
                raise ValueError("Wrong bind mode")

        # sum all features to one resulting image
        # dim=1, dim=features
        image_latent = torch.sum(features, dim=-2)

        decoded_image = self.model.decoder(image_latent)

        title = 'Simple example'
        if multiply_by_placeholders:
            title += ' with binding on placeholders'
        else:
            title += ' without binding on placeholders'
        title += "\n" + ", ".join([
            f'{feature_name}: {self.dataset.possible_values[feature_name][value]:.02f}' for feature_name, value in
            zip(self.dataset.feature_names, feature_values)])

        if display:
            self.visualizer.plot_before_after(image[0], decoded_image[0],
                                              'Image', 'Decoded Image',
                                              title=title)

        return image, labels, features, image_latent, decoded_image






class Visualizer:
    def __init__(self):
        # Logger
        # wandb.init(project="vsa_encoder_stats")
        pass

    def plot_before_after(self,
                          image_before: torch.Tensor,
                          image_after: torch.Tensor,
                          text_before: str,
                          text_after: str,
                          title: str):
        fig, ax = plt.subplots(1, 2)
        plt.suptitle(title)

        for i, (img, title) in enumerate(zip([image_before, image_after], [text_before, text_after])):
            ax[i].imshow(img.permute(1, 2, 0).detach().numpy(), cmap='gray')
            ax[i].tick_params(top=False, bottom=False, left=False, right=False,
                              labelleft=False, labelbottom=False)
            ax[i].set_xlabel(title)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    # add PROGRAM level args
    program_parser = parser.add_argument_group('program')
    program_parser.add_argument("--model_checkpoint_path", type=Path,
                                default='/home/yessense/PycharmProjects/vsa-encoder/checkpoints/epoch=194-step=304590.ckpt')
    program_parser.add_argument("--dataset_path", type=Path,
                                default="/home/yessense/PycharmProjects/vsa-encoder/one_exchange/dsprite_train.npz")

    # parse input
    config = parser.parse_args()

    if isinstance(config, Namespace):
        dict_args = vars(config)
    else:
        dict_args = dict(config)

    stats = Stats(**dict_args)

    # --------------------------------------------------
    # -- Simple display
    # --------------------------------------------------

    # # feature_names ('shape', 'scale', 'orientation', 'posX', 'posY')
    # # features_count [3, 6, 40, 32, 32]
    # stats.simple_process_image([0, 0, 0, 0, 0],
    #                            multiply_by_placeholders=False)
    # # top left
    # stats.simple_process_image([0, 0, 0, 0, 0],
    #                            multiply_by_placeholders=True)
    # # bottom left
    # stats.simple_process_image([0, 0, 0, 0, 31],
    #                            multiply_by_placeholders=True)
    #
    # # top right
    # stats.simple_process_image([0, 0, 0, 31, 0],
    #                            multiply_by_placeholders=True)
    # # bottom right
    # stats.simple_process_image([0, 0, 0, 31, 31],
    #                            multiply_by_placeholders=True)
    #
    # # middle
    # stats.simple_process_image([0, 0, 0, 25, 0],
    #                            multiply_by_placeholders=True)
    # # middle y
    # stats.simple_process_image([0, 0, 0, 25, 25],
    #                            multiply_by_placeholders=True)
    #

    # # oval top right
    # stats.simple_process_image([1, 0, 0, 31, 0],
    #                            multiply_by_placeholders=True)
    # # heart bottom right
    # stats.simple_process_image([2, 0, 0, 31, 31],
    #                            multiply_by_placeholders=True)

    # --------------------------------------------------
    # -- Restore from nth combination of features
    # --------------------------------------------------

    image_idx = random.randint(0, len(stats.dataset))
    stats.restore_from_nth_features(4, image_idx)
    stats.restore_from_nth_features(3, image_idx)
    stats.restore_from_nth_features(2, image_idx)
    stats.restore_from_nth_features(1, image_idx)

    print("Done")
