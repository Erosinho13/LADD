import os
import json
import numpy as np

from dataset import get_dataset
from collections import defaultdict
from torchvision.transforms.functional import normalize


class DatasetHandler(object):

    def __init__(self, args, writer):

        self.args = args
        self.writer = writer
        self.clients_args = {'train': [], 'test': [], 'all_train': []}
        self.source_stats = {}
        self.target_stats = {}

        if self.args.source_dataset != '':
            self.__source_dataset_init()
            self.writer.set_source_img_utils(self.args.target_dataset, mean=self.source_stats['mean'],
                                             std=self.source_stats['std'])
        self.__target_dataset_init()
        self.writer.set_target_img_utils(self.args.target_dataset, mean=self.target_stats['mean'],
                                         std=self.target_stats['std'])

    def __call__(self):
        return self.clients_args

    @staticmethod
    def __get_paths(root, dataset, clients_type):
        if dataset == 'gta5':
            clients_type = ''
        train_data_dir = os.path.join(root, dataset, 'splits', clients_type, 'train')
        test_data_dir = os.path.join(root, dataset, 'splits', clients_type, 'test')
        return train_data_dir, test_data_dir

    @staticmethod
    def __read_dir(data_dir):

        data = defaultdict(lambda: {})

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            data.update(cdata['user_data'])

        return data

    def __read_target_data(self, train_data_dir, test_data_dir):
        train_data = self.__read_dir(train_data_dir)
        test_data = self.__read_dir(test_data_dir)

        return train_data, test_data

    def __preprocess_target_train_data(self, train_data):
        train_data_all = {'x': [], 'y': []}
        for c in train_data.keys():
            train_data_all['x'].extend(train_data[c]['x'])
            if 'y' not in train_data[c]:
                assert 'source' not in c and self.args.target_dataset in ('crosscity',)
                continue
            train_data_all['y'].extend(train_data[c]['y'])
        all_train_data = {'centralized_user': train_data_all}
        if self.args.framework == 'centralized':
            return all_train_data
        return train_data, all_train_data

    def __source_dataset_init(self):

        train_transform, test_transform, dataset = \
            get_dataset(self.args.model, self.args.source_dataset, target_dataset=self.args.target_dataset,
                        cv2=self.args.cv2, random_flip=self.args.random_flip, color_jitter=self.args.color_jitter,
                        gaussian_blur=self.args.gaussian_blur)

        train_ds = dataset(root='data', transform=train_transform, test_transform=test_transform,
                           split='train')
        test_ds = dataset(root='data', transform=train_transform, test_transform=test_transform,
                          split='test')

        self.source_stats = {'mean': train_ds.mean, 'std': train_ds.std}

        self.clients_args['train'].append({'client_id': 'source_train_data', 'dataset': train_ds})
        self.clients_args['test'].append({'client_id': 'source_test_data', 'dataset': test_ds})

    def __gen_ds(self, paths, dataset_name, dataset, train_transform, test_transform, split='train', hp_filtered=False):
        if dataset_name == 'cityscapes':
            return dataset(paths, 'data', transform=train_transform, test_transform=test_transform,
                           hp_filtered=hp_filtered, double=self.args.double_dataset and split == 'train',
                           quadruple=self.args.quadruple_dataset and split == 'train')
        if dataset_name in ['crosscity', 'mapillary']:
            return dataset(paths, root='data', transform=train_transform, test_transform=test_transform,
                           hp_filtered=hp_filtered)
        raise NotImplementedError

    def __target_dataset_init(self):

        dataset_name = self.args.target_dataset
        train_data_dir, test_data_dir = self.__get_paths('data', dataset_name, self.args.clients_type)
        train_data, test_data = self.__read_target_data(train_data_dir, test_data_dir)
        train_data = self.__preprocess_target_train_data(train_data)

        if self.args.framework == 'federated':
            if self.args.centr_fda_ft_uda:
                train_data, all_train_data = train_data[1], train_data[1]
            else:
                train_data, all_train_data = train_data
        else:
            all_train_data = {}

        train_transform, test_transform, dataset = \
            get_dataset(self.args.model, dataset_name, self.args.double_dataset, self.args.quadruple_dataset,
                        cv2=self.args.cv2)
        train_users, test_users = train_data.keys(), test_data.keys()

        for users, split_data, split in zip((train_users, test_users), (train_data, test_data), ('train', 'test')):

            for i, user in enumerate(users):

                ds = self.__gen_ds(split_data[user], dataset_name, dataset, train_transform, test_transform,
                                   split=split, hp_filtered=self.args.hp_filtered)

                if not self.target_stats:
                    self.target_stats = {'mean': ds.mean, 'std': ds.std}

                self.clients_args[split].append({'client_id': user, 'dataset': ds})

        if self.args.framework == 'federated':
            ds = self.__gen_ds(all_train_data['centralized_user'],
                               dataset_name, dataset, train_transform, test_transform, split='test')
            self.clients_args['all_train'].append({'client_id': 'all_target_train_data', 'dataset': ds})


class Denormalize(object):

    def __init__(self, mean, std):
        if mean is not None and std is not None:
            self._mean = -np.array(mean) / std
            self._std = 1 / np.array(std)

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def color_map(dataset):

    if dataset == 'cityscapes' or dataset == 'mapillary':
        color = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                 (0, 0, 0)]
        n = 256
    elif dataset == 'crosscity':
        color = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (250, 170, 30), (220, 220, 0), (107, 142, 35),
                 (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 60, 100), (0, 0, 230), (119, 11, 32),
                 (0, 0, 0)]
        n = 256
    else:
        raise NotImplementedError

    cmap = np.zeros((n, 3), dtype='uint8')
    for i, co in enumerate(color):
        cmap[i] = co
    return cmap.astype(np.uint8)


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]
