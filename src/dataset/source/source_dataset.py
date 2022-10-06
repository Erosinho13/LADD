import os
import torch
import numpy as np

from torch import from_numpy
from abc import ABCMeta, abstractmethod
from torchvision.datasets import VisionDataset
from ..load_img import load_img


class SourceDataset(VisionDataset, metaclass=ABCMeta):

    def __init__(self, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, split='train'):

        super().__init__(root, transform=transform, target_transform=None)

        self.root = root
        self.test = split not in ('train',)
        self.transform = transform
        self.test_transform = test_transform
        self.mean = mean
        self.std = std
        self.cv2 = cv2

        self.target_transform = self.__map_labels()
        self.paths = []

        self.return_unprocessed_image = False
        self.style_tf_fn = None

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    def __getitem__(self, index):

        transform = self.transform if not self.test else self.test_transform
        x_path, y_path = self.paths[index].strip('\n').split(' ')
        x, y, _ = load_img(
            x_path=os.path.join(self.root, self.dataset_dir, 'data', x_path.lstrip("/")),
            y_path=os.path.join(self.root, self.dataset_dir, 'data', y_path.lstrip("/")),
            cv2=self.cv2
        )
        if self.return_unprocessed_image:
            return x
        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)
        x, y = transform(x, y)
        y = self.target_transform(y)

        if self.test:
            plot_x = torch.clone(x)
            return (plot_x, x), y

        return x, y

    def __len__(self):
        return len(self.paths)

    def __map_labels(self):
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in self.labels2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])

    @property
    @abstractmethod
    def labels2train(self):
        pass

    @property
    @abstractmethod
    def images_dir(self):
        pass

    @property
    @abstractmethod
    def target_dir(self):
        pass

    @property
    @abstractmethod
    def dataset_dir(self):
        pass
