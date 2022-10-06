import torch
import random
import importlib
import numpy as np

from .dist_utils import initialize_distributed
from .writer import Writer, CustomWandbLogger


def setup_env(args):
    # ===== Setup Writer ===== #
    writer = Writer(args.local_rank)

    # ===== Setup distributed ===== #
    writer.write("Setting up distributed...")
    device, rank, world_size = initialize_distributed(args.local_rank)
    writer.write(f"Let's use {args.n_devices} GPUs!")
    writer.write(f"Done")

    # ===== Initialize wandb ===== #
    writer.write("Initializing wandb...")
    wandb_logger = CustomWandbLogger(args)
    writer.set_wandb(wandb_logger)
    writer.wandb.log_hyperparams(args)
    writer.write("Done.")

    # ===== Setup random seed to reproducibility ===== #
    if args.random_seed is not None:
        writer.write("Setting up the random seed for reproducibility...")
        set_seed(args.random_seed)
        writer.write("Done.")

    return writer, device, rank, world_size


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dynamic_import(framework, fw_task, parent_class):

    module_name = framework if parent_class != 'client' else 'clients'
    prefix = ''.join([token.title() for token in fw_task.split('_')])
    class_name = f'{prefix}{parent_class.title()}' if fw_task != 'ladd' else f'LADD{parent_class.title()}'
    trainer_module = importlib.import_module(module_name)
    trainer_class = getattr(trainer_module, class_name)

    return trainer_class
