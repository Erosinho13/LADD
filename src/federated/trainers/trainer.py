import copy
import torch

from utils import dynamic_import, weight_train_loss
from general_trainer import GeneralTrainer


class Trainer(GeneralTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)
        self.all_target_client = self.gen_all_target_client()

    def gen_all_target_client(self):
        client_class = dynamic_import(self.args.framework, self.args.fw_task, 'client')
        cl_args = {**self.clients_shared_args, **self.clients_args['all_train'][0]}
        return client_class(**cl_args, batch_size=self.args.test_batch_size, test_user=True)

    def server_setup(self):
        server_class = dynamic_import(self.args.framework, self.args.fw_task, 'server')
        server = server_class(self.model, self.writer, self.args.local_rank, self.args.server_lr,
                              self.args.server_momentum, self.args.server_opt, self.args.source_dataset)
        return server

    @staticmethod
    def set_metrics(writer, num_classes):
        raise NotImplementedError

    def handle_ckpt_step(self):
        raise NotImplementedError

    def get_optimizer_and_scheduler(self):
        return None, None

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.server.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        if "server_optimizer_state" in self.checkpoint.keys():
            self.server.optimizer.load_state_dict(self.checkpoint["server_optimizer_state"])
            self.writer.write(f"[!] Server optimizer restored.")

    def save_model(self, step, optimizer=None, scheduler=None):
        state = {
            "step": step,
            "model_state": self.server.model_params_dict
        }
        if self.server.optimizer is not None:
            state["server_optimizer_state"] = self.server.optimizer.state_dict()
        torch.save(state, self.ckpt_path)
        self.writer.wandb.save(self.ckpt_path)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def plot_train_metric(self, r, metric, losses, plot_metric=True):
        if self.args.local_rank == 0:
            round_losses = weight_train_loss(losses)
            self.writer.plot_step_loss(metric.name, r, round_losses)
            if plot_metric:
                self.writer.plot_metric(r, metric, '', self.ret_score)
