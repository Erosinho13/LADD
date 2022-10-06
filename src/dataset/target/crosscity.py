import os

from .target_dataset import TargetDataset


class CrossCity(TargetDataset):

    task = 'segmentation'
    ds_type = 'unsupervised'
    labels2train = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6, 24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}
    images_dir = os.path.join('crosscity', 'data', 'cities')
    target_dir = os.path.join('crosscity', 'data', 'cities')

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, hp_filtered=False):
        super().__init__(paths, root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         hp_filtered=hp_filtered)
