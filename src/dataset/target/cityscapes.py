import os
import dataset.transform as transform_pytorch
import dataset.transform_cv2 as transform_cv2

from .target_dataset import TargetDataset


class Cityscapes(TargetDataset):

    task = 'segmentation'
    ds_type = 'supervised'
    labels2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                    27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    images_dir = os.path.join('cityscapes', 'data', 'leftImg8bit')
    target_dir = os.path.join('cityscapes', 'data', 'gtFine')

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, double=False, quadruple=False, hp_filtered=False):

        super().__init__(paths, root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         hp_filtered=hp_filtered)

        self.true_len = len(self.paths['x'])
        self.double = double
        self.quadruple = quadruple
        self.tr = transform_pytorch if not cv2 else transform_cv2

    def _preprocess_index(self, index):
        if index >= self.true_len and (self.double or self.quadruple):
            index %= self.true_len
        return index

    def _preprocess_images(self, x, y, original_index):
        if (self.double and original_index >= self.true_len) or (self.quadruple and original_index >= 2*self.true_len):
            x, y = self.tr.RandomHorizontalFlip(1)(x, y)
        return x, y

    def __len__(self):
        if self.test:
            return self.true_len
        if self.double:
            return 2*self.true_len
        if self.quadruple:
            return 4*self.true_len
        return self.true_len
