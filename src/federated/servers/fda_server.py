import copy
import torch
import torch.nn as nn
import os
import numpy as np

from .oracle_server import OracleServer
from torch.utils import data
from utils import StyleAugment, DistributedRCSSampler


class FdaServer(OracleServer):

    def __init__(self, args, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer, source_dataset=source_dataset)

        self.args = args
        self.styleaug = StyleAugment(args.n_images_per_style, args.fda_L, args.fda_size, b=args.fda_b)
        self.writer.write('Extracting styles...')
        self.styleaug.add_style(self.source_dataset)
        self.writer.write('Done')
        self.batch_norm_dict = {"server": None, "client": []}

    def _load_silo_batch_norm_stats(self, model, node="client", i=0):
        if len(self.batch_norm_dict[node]) - 1 >= i:
            state_dict = model.state_dict()
            for name, layer in copy.deepcopy(model).named_modules():
                if isinstance(layer, nn.BatchNorm2d):
                    state_dict[name + ".running_mean"] = self.batch_norm_dict[node][i][name]["running_mean"]
                    state_dict[name + ".running_var"] = self.batch_norm_dict[node][i][name]["running_var"]
            model.load_state_dict(state_dict)

    def _save_silo_batch_norm_stats(self, model, node="client"):
        batch_norm_dict = {}
        for name, layer in copy.deepcopy(model).named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                batch_norm_dict[name] = {
                    "running_mean": layer.running_mean,
                    "running_var": layer.running_var
                }
        self.batch_norm_dict[node].append(batch_norm_dict)

    def save_pretrained_server(self, model_state, best: bool):
        print("Saving best server pretrained ckpt...")
        state = {"model_state": model_state}
        ckpt_path = os.path.join('checkpoints', self.args.framework, self.args.source_dataset, self.args.target_dataset,
                                 f'pretrained_server{"_best" if best else ""}.ckpt')
        torch.save(state, ckpt_path)
        self.writer.wandb.save(ckpt_path)
        print("Done.")

    def train_source(self, train_clients, test_clients, train_metric, test_metric, optimizer, scheduler,
                     ret_score, device, test_fn, num_epochs=None, num_steps=None):

        train_clients[0].model.load_state_dict(self.model_params_dict)

        max_scores = [0] * len(test_clients)

        num_source_epochs = self.args.num_source_epochs if num_epochs is None else num_epochs
        if num_steps is not None:
            stop_at_step = num_steps % len(train_clients[0].loader)
            num_source_epochs = num_steps // len(train_clients[0].loader) + min(1, stop_at_step)

        for e in range(0, num_source_epochs):

            self.writer.write(f"EPOCH: {e + 1}/{num_source_epochs}")
            train_clients[0].model.train()
            _ = train_clients[0].run_epoch(e, optimizer, train_metric, scheduler, e_name='EPOCH',
                                           stop_at_step=num_steps if (e + 1) == num_source_epochs and num_steps is
                                           not None and num_steps > 0 else None)
            train_metric.synch(device)
            self.writer.plot_metric(e, train_metric, str(train_clients[0]), ret_score)
            train_metric.reset()

            if (e + 1) % self.args.server_test_interval == 0 or (e + 1) == num_source_epochs:
                max_scores, found_new_max = test_fn(test_clients, test_metric, e, 'EPOCH', max_scores, cl_type='target')
                if found_new_max:
                    self.save_pretrained_server(train_clients[0].model.state_dict(), best=True)

        self.model_params_dict = copy.deepcopy(train_clients[0].model.state_dict())

        return max_scores

    def count_classes(self):
        self.writer.write("Extracting pseudo labels stats...")
        for i, c in enumerate(self.selected_clients):
            self.writer.write(f"client {i + 1}/{len(self.selected_clients)}: {c}...")
            class_probs, class_by_image = c.count_classes()
            c.loader = data.DataLoader(c.dataset, batch_size=c.batch_size, worker_init_fn=c.seed_worker,
                                       sampler=DistributedRCSSampler(c.dataset, num_replicas=c.world_size,
                                                                     rank=c.rank, class_probs=class_probs,
                                                                     class_by_image=class_by_image,
                                                                     seed=self.args.random_seed),
                                       num_workers=4*c.num_gpu, drop_last=True, pin_memory=True)
        self.writer.write("Done.")

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):

            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")

            c.model.load_state_dict(self.model_params_dict)
            if self.args.silobn:
                self._load_silo_batch_norm_stats(c.model, node="client", i=i)

            out = c.train(partial_metric, r=r)

            if self.args.silobn:
                self._save_silo_batch_norm_stats(c.model, node="client")

            if self.local_rank == 0:
                num_samples, update, dict_losses_list = out
                losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
            else:
                num_samples, update = out

            if self.optimizer is not None:
                update = self._compute_client_delta(update)

            self.updates.append((num_samples, update))

        if self.local_rank == 0:
            return losses
        return None

    @staticmethod
    def clients_city(possible_clients, city):
        return [client for client in possible_clients if city in client.id]

    def select_clients(self, my_round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        if self.args.n_clients_per_city and self.args.target_dataset == "crosscity":
            clients = []
            for city in ["Taipei", "Rome", "Rio", "Tokyo"]:
                clients.append(np.random.permutation(self.clients_city(possible_clients, city))[:num_clients])
            self.selected_clients = np.random.permutation([item for sublist in clients for item in sublist])
            self.writer.write(f"selected clients: {[client.id for client in self.selected_clients]}, "
                              f"unique: {len(set(self.selected_clients))}")
        else:
            self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def update_clients_lr(self, lr):
        for c in self.selected_clients:
            c.lr_fed = lr
