import torch

from general_trainer import GeneralTrainer
from utils import get_optimizer_and_scheduler


class Trainer(GeneralTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    def server_setup(self):
        return None

    @staticmethod
    def set_metrics(writer, num_classes):
        raise NotImplementedError

    def handle_ckpt_step(self):
        raise NotImplementedError

    def max_iter(self):
        raise NotImplementedError

    def get_optimizer_and_scheduler(self):
        return get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter())

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state"])
        self.writer.write(f"[!] Optimizer and scheduler restored.")

    def save_model(self, step, optimizer=None, scheduler=None):
        state = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        if scheduler is not None:
            state["scheduler_state"] = scheduler.state_dict()
        torch.save(state, self.ckpt_path)
        self.writer.wandb.save(self.ckpt_path)

    def train(self, *args, **kwargs):
        raise NotImplementedError
