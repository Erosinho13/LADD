import os
import sys
import time
import warnings

from utils import dynamic_import
from utils import dist_utils, setup_env
from utils import parse_args, modify_command_options, check_args


def run_experiment():
    writer, device, rank, world_size = setup_env(args)

    trainer_class = dynamic_import(args.framework, args.fw_task, 'trainer')
    trainer = trainer_class(args, writer, device, rank, world_size)

    writer.write("The experiment begins...")
    max_score = trainer.train(*trainer.train_args, **trainer.train_kwargs)
    writer.write("Training completed.")

    if trainer.model.module.task == 'classification':
        writer.write(f"Final Overall Acc: {round(max_score[0] * 100, 3)}%")
    elif trainer.model.module.task == 'segmentation':
        writer.write(f"Final mIoU: {round(max_score[0] * 100, 3)}%")
    else:
        raise NotImplementedError

    dist_utils.cleanup()


if __name__ == '__main__':

    sys.setrecursionlimit(10000)

    os.chdir('..')

    start = time.time()

    parser = parse_args()
    args = parser.parse_args()
    args = modify_command_options(args)
    check_args(args)

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    run_experiment()

    end = time.time()

    secs = end - start
    mins = secs // 60
    secs %= 60
    hours = mins // 60
    mins %= 60

    if args.local_rank == 0:
        print(f"Elapsed time: {int(hours)}h, {int(mins)}min, {round(secs, 2)}s")
