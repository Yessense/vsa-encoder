import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import torchvision.io.image
from torchvision.io import read_image


class PairedCogentClevr(Dataset):
    def __init__(self, scenes_dir, images_dir, img_shape=(3, 128, 128), max_objects=10, for_stats=False):
        self.scenes_dir = scenes_dir
        self.images_dir = images_dir
        self.img_shape = img_shape
        self.max_objects = max_objects

        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self._size = len(self.scenes_list)
        self.features = ['shape', 'color', 'size', 'material', 'x', 'y']
        self.features_size: int = 6

        self.img_template = "{name}_{idx:06d}.png"
        self.scene_template = "scene_{name}_{idx:06d}.png"
        self.obj_template = "obj{n}_{idx:06d}.png"

        self.json_template = "scene_{idx:06d}.json"
        self.for_stats = for_stats

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        # Open json file
        scene = self.scenes_list[idx]
        with open(os.path.join(self.scenes_dir, scene), 'r') as f:
            scenes_list = json.load(f)
        scenes_dict = {row['image_filename']: row for row in scenes_list}

        # Load discrete image, pair and scenes
        image_name = self.img_template.format(name='img', idx=idx)
        image = self.get_image(image_name)
        img_scene = self.get_image(self.scene_template.format(name='img', idx=idx))

        pair_name = self.img_template.format(name='pair', idx=idx)
        pair_image = self.get_image(pair_name)
        pair_scene = self.get_image(self.scene_template.format(name='pair', idx=idx))

        # Load other objects
        # Add masks to multiply on unused objects in model later on
        obj_images = []
        obj_masks = torch.zeros(self.max_objects - 1, dtype=bool)

        # max objects = 10, minus (img, pair) = 2
        for obj_idx in range(1, self.max_objects - 2 + 1):
            obj_name = self.obj_template.format(n=obj_idx, idx=idx)
            if obj_name in scenes_dict:
                obj_images.append(self.get_image(obj_name))
                obj_masks[obj_idx - 1] = True
            else:
                obj_images.append(torch.zeros(self.img_shape))

        exchange_labels = self.get_difference(scenes_dict[image_name]['objects'][0],
                                              scenes_dict[pair_name]['objects'][0])
        obj_images = torch.stack(obj_images, dim=0)

        if not self.for_stats:
            return exchange_labels, image, pair_image, img_scene, pair_scene, obj_images, obj_masks
        else:
            return image, self.get_labels(scenes_dict[image_name]['objects'][0])

    def get_image(self, image_name):
        img_path = os.path.join(self.images_dir, image_name)
        img = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB) / 255
        return img

    def get_difference(self, obj1, obj2):
        exchange_labels = torch.zeros(self.features_size, dtype=bool)

        if obj1['shape'] != obj2['shape']:
            exchange_labels[0] = True
        if obj1['color'] != obj2['color']:
            exchange_labels[1] = True
        if obj1['size'] != obj2['size']:
            exchange_labels[2] = True
        if obj1['material'] != obj2['material']:
            exchange_labels[3] = True
        if obj1['3d_coords'][0] != obj2['3d_coords'][0]:
            exchange_labels[4] = True
        if obj1['3d_coords'][1] != obj2['3d_coords'][1]:
            exchange_labels[5] = True
        # else:
        #     raise ValueError(f'All features are the same {obj1}, {obj2}')
        return exchange_labels.unsqueeze(-1)

    def get_labels(self, obj):
        labels = []
        labels.append(obj['shape'])
        labels.append(obj['color'])
        labels.append(obj['size'])
        labels.append(obj['material'])
        labels.append(obj['3d_coords'][0])
        labels.append(obj['3d_coords'][1])
        return labels


if __name__ == '__main__':
    dataset = PairedCogentClevr(scenes_dir='/home/yessense/projects/clevr_multi_cogent/test/scenes',
                                images_dir='/home/yessense/projects/clevr_multi_cogent/test/images',
                                img_shape=(3, 128, 128), for_stats=False)

    def plot_one_example(batch):
        exchange_labels, image, pair_image, img_scene, pair_scene, obj_images, obj_masks = batch

        names = ['Изображение', 'Донор', 'Исх. cцена', 'Изм. сцена']
        images = [image, pair_image, img_scene, pair_scene]
        features = ['shape', 'color', 'size', 'material', 'x', 'y']

        n_objects = sum(obj_masks)
        max_plots = max(4, n_objects)
        fig, ax = plt.subplots(2, max_plots)

        title = "Different features: "
        title += ", ".join([features[i] for i in range(len(features)) if exchange_labels[i]])
        plt.suptitle(f'{title}')

        for i, (name, img) in enumerate(zip(names, images)):
            ax[0, i].imshow(img.permute(1, 2, 0))
            ax[0, i].tick_params(top=False, bottom=False, left=False, right=False,
                                 labelleft=False, labelbottom=False)
            ax[0, i].set_xlabel(names[i])

        for i in range(max_plots):
            if i < n_objects:
                ax[1, i].imshow(obj_images[i].permute(1, 2, 0))
                ax[1, i].tick_params(top=False, bottom=False, left=False, right=False,
                                     labelleft=False, labelbottom=False)
                ax[1, i].set_xlabel(f'Объект {i}')
            else:
                ax[1, i].set_axis_off()

        fig.tight_layout()
        plt.show()


    for i in range(3):
        plot_one_example(dataset[i])

    print("Done")
