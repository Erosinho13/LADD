import os
import gzip
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch import from_numpy
from torch.utils import data
from torchvision import transforms as tr
from ..load_img import load_img
from abc import ABCMeta, abstractmethod


class TargetDataset(data.Dataset, metaclass=ABCMeta):

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, hp_filtered=False):

        self.paths = paths
        self.root = root
        self.transform = transform
        self.test_transform = test_transform
        self.mean = mean if self.task == 'segmentation' else None
        self.std = std if self.task == 'segmentation' else None
        self.cv2 = cv2
        self.test = False
        self.style_tf_fn = None
        self.return_unprocessed_image = False
        self.hp_filtered = hp_filtered

        if self.task == 'segmentation':
            self.target_transform = self.__map_labels()

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    def __getitem__(self, index):
        index = self._preprocess_index(index)
        x, y, x_hpf = self._get_images(index)

        if self.return_unprocessed_image:
            return x

        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)

        if self.test:
            plot_x, x, y = self._apply_test_transform(x, y, x_hpf=x_hpf)
            if self.hp_filtered:
                x, x_hpf = x
                return (plot_x, x, x_hpf), y
            return (plot_x, x), y

        return self._apply_train_transform(x, y, x_hpf)

    def __len__(self):
        return len(self.paths['x'])

    def __map_labels(self):
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in self.labels2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])

    @staticmethod
    def _preprocess_index(index):
        return index

    def _get_images(self, index):
        x_path = os.path.join(self.root, self.images_dir, self.paths['x'][index])
        x_hpf_path = x_path.replace('data', 'hp_filtered') if self.hp_filtered else None

        if self.ds_type == 'supervised' or self.test:
            try:
                y_path = os.path.join(self.root, self.target_dir, self.paths['y'][index])
            except IndexError:
                y_path = None
        else:
            y_path = None
        x, y, x_hpf = load_img(x_path=x_path, y_path=y_path, cv2=self.cv2, x_hpf_path=x_hpf_path)
        return x, y, x_hpf

    @staticmethod
    def _preprocess_images(x, y, original_index):
        return x, y

    def _apply_test_transform(self, x, y, x_hpf=None):
        if x_hpf is not None:
            x_hpf = tr.Resize((x.size[1], x.size[0]))(x_hpf)
            x, x_hpf = self.test_transform(x, x_hpf)

            return x, (x, x_hpf), self.target_transform(y)
        return self.test_transform(x), self.test_transform(x), self.target_transform(y)

    def _apply_train_transform(self, x, y, x_hpf=None):
        if self.ds_type == 'supervised':
            x, y = self.transform(x, y)
            y = self.target_transform(y)
            return x, y
        if x_hpf is not None:
            return self.transform(x, x_hpf)
        return self.transform(x)

    def generate_fft(self, size=(1024, 512)):

        output_file = \
            f'{os.path.join(self.root, os.path.split(self.images_dir)[0], f"amp_{self.__class__.__name__}")}.npy.gz'
        ftt_data = {}
        for index in tqdm(range(len(self.paths['x']))):
            x_path = os.path.join(self.root, self.images_dir, self.paths['x'][index])
            pil_img, _ = load_img(x_path)
            pil_img = pil_img.resize(size, Image.BICUBIC)
            img_np = np.asarray(pil_img, np.float32)
            img_np = img_np[:, :, ::-1]
            img_np = img_np.transpose((2, 0, 1))
            fft_np = np.fft.fft2(img_np, axes=(-2, -1))
            amp_np = np.abs(fft_np)
            ftt_data[self.paths['x'][index]] = np.asarray(amp_np, np.float16)

        f = gzip.GzipFile(output_file, "w")
        np.save(f, ftt_data)
        f.close()

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def ds_type(self):
        pass

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
