import itertools
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, List
import seaborn as sns

import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vsa import bind, unbind, sim

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

        self.codebook = [np.zeros((self.dataset.features_count[i], self.model.latent_dim)) for i in
                         range(self.model.n_features)]
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

        if display:
            title = 'Simple example'
            if multiply_by_placeholders:
                title += ' with binding on placeholders'
            else:
                title += ' without binding on placeholders'
            title += "\n" + ", ".join([
                f'{feature_name}: {self.dataset.possible_values[feature_name][value]:.02f}' for feature_name, value in
                zip(self.dataset.feature_names, feature_values)])
            self.visualizer.plot_before_after(image[0], decoded_image[0],
                                              'Image', 'Decoded Image',
                                              title=title)

        return image, labels, features, image_latent, decoded_image

    def decode_from_codebook(self, feature_values, display: bool=True):
        # feature_names ('shape', 'scale', 'orientation', 'posX', 'posY')
        # features_count [3, 6, 40, 32, 32]
        image_idx: int = self.dataset._get_element_pos(feature_values)
        image, labels = self.dataset[image_idx]
        image = torch.from_numpy(image).to(self.device).float()
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        codebook_features = [self.codebook[i][val] for i, val in enumerate(feature_values)]
        codebook_features = torch.tensor(codebook_features).to(self.device).float()
        codebook_features = torch.unsqueeze(codebook_features, 0)
        image_latent = torch.sum(codebook_features, dim=-2)

        decoded_image = self.model.decoder(image_latent)

        if display:
            title = 'Decoded image from a codebook features combo'
            title += "\n" + ", ".join([
                f'{feature_name}: {self.dataset.possible_values[feature_name][value]:.02f}' for feature_name, value in
                zip(self.dataset.feature_names, feature_values)])
            self.visualizer.plot_before_after(image[0], decoded_image[0],
                                              'Image', 'Decoded Image',
                                              title=title)

        return image, labels, codebook_features, image_latent, decoded_image

    def get_latents(self, num_latents: int = 1000, draw_latents: bool = False, multiply_by_placeholders: bool = True):
        """Encode all images in dataset"""
        batch_size = 10
        features_list = []
        labels_list = []
        dataloader = self.make_dataloader(batch_size=batch_size)

        for i, (images, labels) in tqdm(zip(range(num_latents // batch_size), dataloader)):
            images = images.to(self.model.device).unsqueeze(1).float()
            if multiply_by_placeholders:
                features, _, _ = self.model.encode(images)
            else:
                features = self.model.get_features(images)
            features = features.cpu().detach().numpy()
            features_list.append(features)

            labels = labels.numpy()
            labels_list.append(labels)

        features_list = np.concatenate(features_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        if draw_latents:

            print(f'Means of each feature')
            for i, name in enumerate(self.dataset.feature_names):
                print(f'{name}. mean: {features_list[:, i, :].mean():0.4f}, std: {features_list[:, i, :].std():0.4f}')
            for feature_number in range(self.model.n_features):
                self.visualizer.show_feature_mean_vectors(features_list, labels_list, feature_number,
                                                          self.dataset.features_count, self.dataset.feature_names,
                                                          show_only_target_feature_vector=True)

                self.visualizer.show_feature_mean_vectors(features_list, labels_list, feature_number,
                                                          self.dataset.features_count, self.dataset.feature_names,
                                                          show_only_target_feature_vector=False)
        return features_list, labels_list

    def set_codebook(self, num_latents: int = 10_000):
        features_list, labels_list = self.get_latents(num_latents=num_latents,
                                                      draw_latents=False,
                                                      multiply_by_placeholders=True)
        codebook = [np.zeros((self.dataset.features_count[i], self.model.latent_dim)) for i in
                    range(self.model.n_features)]

        for feature_number in range(self.model.n_features):
            for feature_index in range(self.dataset.features_count[feature_number]):
                codebook[feature_number][feature_index] = np.mean(
                    features_list[:, feature_number][labels_list[:, feature_number] == feature_index], axis=0)

        self.codebook = codebook

    def check_vsa(self, feature_values: List):

        image, labels, features, image_latent, decoded_image = self.simple_process_image(feature_values,
                                                                                         multiply_by_placeholders=True,
                                                                                         display=True)

        hd_placeholders = self.model.hd_placeholders.data.squeeze(0)

        for i, feature_vector in enumerate(features[0]):
            output_vec = unbind(image_latent[0], feature_vector)

            similarities = []
            for j, hd_placeholder in enumerate(hd_placeholders):
                similarity = sim(output_vec, hd_placeholder)
                similarities.append(similarity)
                print(f'Feature unbinded {i} is similar to feature {j} for {similarity}')

            similarities = torch.stack(similarities, dim=0)
            max_pos = torch.argmax(similarities, dim=0)
            print(f'Feature unbinded {i} is most similar to feature {max_pos}')

            # print("cs")

        return None


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

    def show_feature_mean_vectors(self, features: np.ndarray, labels: np.ndarray, feature: int,
                                  features_count,
                                  feature_names,
                                  show_only_target_feature_vector: bool = False):

        feature_name = feature_names[feature]
        title = f'Mean vector for feature {feature_name}\n'

        if show_only_target_feature_vector:
            vectors = features[:, feature, :]
            title += f'Before sum'
        else:
            vectors = np.sum(features, axis=1)
            title += f'After sum'

        mean_vectors = []

        for i in range(features_count[feature]):
            mean_vectors.append(np.mean(vectors[labels[:, feature] == i], axis=0))

        fig, ax = plt.subplots(figsize=(30, 8))
        mean_vectors = np.array(mean_vectors)
        sns.heatmap(mean_vectors, ax=ax, center=0)
        fig.suptitle(title)
        ax.set_ylabel(f'{feature_name}')
        ax.set_xlabel('coordinate')
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

    # feature_names ('shape', 'scale', 'orientation', 'posX', 'posY')
    # features_count [3, 6, 40, 32, 32]
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
    # stats.restore_from_nth_features(4, image_idx)
    # stats.restore_from_nth_features(3, image_idx)
    stats.restore_from_nth_features(2, image_idx)
    stats.restore_from_nth_features(1, image_idx)

    # --------------------------------------------------
    # -- Get latents
    # --------------------------------------------------

    # latents = stats.get_latents(num_latents=1000, draw_latents=True, multiply_by_placeholders=True)
    # latents = stats.get_latents(num_latents=1000, draw_latents=True, multiply_by_placeholders=False)
    # latents = stats.get_latents(num_latents=10000, draw_latents=True, multiply_by_placeholders=True)

    # --------------------------------------------------
    # -- Check vsa
    # --------------------------------------------------

    # processed_latents = stats.check_vsa([2, 2, 2, 24, 0])

    # --------------------------------------------------
    # -- Codebook of mean vectors
    # --------------------------------------------------

    stats.set_codebook(10_000)
    # stats.decode_from_codebook([0, 0, 0, 0, 0])
    # stats.decode_from_codebook([1, 0, 0, 0, 0])
    # stats.decode_from_codebook([2, 0, 0, 0, 0])
    #
    # stats.decode_from_codebook([0, 0, 0, 31, 0])
    # stats.decode_from_codebook([1, 0, 0, 31, 0])
    # stats.decode_from_codebook([2, 0, 0, 31, 0])
    #
    # stats.decode_from_codebook([0, 1, 0, 31, 31])
    # stats.decode_from_codebook([1, 0, 0, 31, 31])
    # stats.decode_from_codebook([2, 0, 0, 31, 31])

    stats.decode_from_codebook([1, 1, 0, 15, 15])
    stats.decode_from_codebook([1, 3, 0, 15, 15])
    stats.decode_from_codebook([1, 4, 0, 15, 15])

    stats.decode_from_codebook([1, 1, 10, 15, 15])
    stats.decode_from_codebook([1, 3, 20, 15, 15])
    stats.decode_from_codebook([1, 4, 30, 15, 15])

    print("Done")
