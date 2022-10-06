import torch
import random
from torch import distributed
from torch.utils.data import DistributedSampler

def initialize_distributed(local_rank):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(local_rank)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    return device, rank, world_size


def cleanup():
    distributed.destroy_process_group()


class DistributedRCSSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas, rank, class_probs, class_by_image, seed):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed)
        self.dataset = dataset
        self.class_probs = class_probs
        self.class_by_image = class_by_image
        random.seed(seed + self.epoch)

    def __iter__(self):
        indices = []
        while len(indices) < self.num_samples:
            random_class = random.choices(list(self.class_probs.keys()), list(self.class_probs.values()))[0]
            random_index = random.choice(self.class_by_image[random_class])
            indices.append(random_index)
        assert len(indices) == self.num_samples
        return iter(indices)
