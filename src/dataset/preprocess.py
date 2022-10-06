from functools import partial

from . import transform as transform_pytorch
from . import transform_cv2 as transform_cv2
from .target import Cityscapes, CrossCity, Mapillary
from .source import GTA5


def to_tens_and_norm(tr, mean, std, cv2):
    if not cv2:
        return [tr.ToTensor(), tr.Normalize(mean=mean, std=std)]
    return [tr.ToTensor(mean=mean, std=std)]


def get_dataset(model_name, dataset_name, double_dataset=None, quadruple_dataset=None, target_dataset=None, cv2=False,
                random_flip=False, color_jitter=False, gaussian_blur=False):

    tr = transform_pytorch if not cv2 else transform_cv2

    if dataset_name == 'cityscapes':

        if model_name == 'deeplabv3':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            dataset = partial(Cityscapes, mean=mean, std=std, cv2=cv2)

            train_transform = [
                tr.RandomScale((0.7, 2)),
                tr.RandomCrop((512, 1024)),
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=std),
            ]

            if not double_dataset and not quadruple_dataset:
                train_transform = [tr.RandomHorizontalFlip(0.5), *train_transform]

            test_transform = to_tens_and_norm(tr, mean, std, cv2)

        else:
            raise NotImplementedError

    elif dataset_name == 'gta5':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        dataset = partial(GTA5, mean=mean, std=std, cv2=cv2, target_dataset=target_dataset)

        train_transform = [
            tr.RandomScale((0.7, 2)),
            tr.RandomCrop((512, 1024), pad_if_needed=True),
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=std),
        ]

        test_transform = to_tens_and_norm(tr, mean, std, cv2)

    elif dataset_name == 'crosscity':

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        dataset = partial(CrossCity, mean=mean, std=std, cv2=cv2)

        train_transform = [
            tr.RandomScale((0.7, 2)),
            tr.RandomCrop((512, 1024), pad_if_needed=True),
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=std),
        ]

        if not double_dataset and not quadruple_dataset:
            train_transform = [tr.RandomHorizontalFlip(0.5), *train_transform]

        test_transform = to_tens_and_norm(tr, mean, std, cv2)

    elif dataset_name == 'mapillary':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        dataset = partial(Mapillary, mean=mean, std=std, cv2=cv2)
        train_transform = [
            tr.FixedResize(width=1024),
            tr.RandomCrop((512, 1024), pad_if_needed=True),
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=std),
        ]
        if not double_dataset and not quadruple_dataset:
            train_transform = [tr.RandomHorizontalFlip(0.5), *train_transform]
        test_transform = [
            tr.FixedResize(width=1024),
            *to_tens_and_norm(tr, mean, std, cv2)
        ]

    else:
        raise NotImplementedError

    if random_flip:
        train_transform[0:0] = [tr.RandomHorizontalFlip(0.5)]

    if gaussian_blur:
        train_transform[-2:-2] = [tr.GaussianBlur()]
    if color_jitter:
        train_transform[-2:-2] = [tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)]

    train_transform = tr.Compose(train_transform)
    test_transform = tr.Compose(test_transform)

    return train_transform, test_transform, dataset
