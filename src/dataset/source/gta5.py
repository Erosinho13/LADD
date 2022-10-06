import os

from dataset.source.source_dataset import SourceDataset


class GTA5(SourceDataset):
    labels2train = {
        'crosscity': {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6, 24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12},
        'cityscapes': {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18},
        'mapillary': {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    }
    images_dir = 'gta5/data/images'
    target_dir = 'gta5/data/labels'
    dataset_dir = 'gta5'
    ds_type = 'supervised'

    def __init__(self, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), cv2=False,
                 split='train', target_dataset=None):
        assert target_dataset in GTA5.labels2train, \
            f'Class mapping missing for {target_dataset}, choose from: {GTA5.labels2train.keys()}'
        self.labels2train = GTA5.labels2train[target_dataset]

        super().__init__(root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         split=split)

        item_list_filepath = os.path.join(root, self.dataset_dir, 'splits', f'{split}.txt')
        self.paths = [ids for ids in open(item_list_filepath)]
