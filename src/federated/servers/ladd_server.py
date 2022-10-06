from .fda_inv_server import FdaInvServer
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os


class LADDServer(FdaInvServer):

    def __init__(self, args, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super().__init__(args, model, writer, local_rank, lr, momentum, optimizer, source_dataset=source_dataset)
        self.clusters_models = []
        self.num_clusters = None
        self.update_counter = 0
        self.global_aggregation = True
        self.encoder_names = []
        self.decoder_names = []
        self.batch_norm_names = []
        non_parameters_names = []
        for name, layer in copy.deepcopy(self.model).named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                self.batch_norm_names.append(name)
            elif "backbone" in name and isinstance(layer, nn.Conv2d):
                self.encoder_names.append(name)
            elif "classifier" in name and isinstance(layer, nn.Conv2d):
                self.decoder_names.append(name)
            else:
                non_parameters_names.append(layer)

        if args.algorithm == "FedAvg":
            self.train_clients = super().train_clients
            self._aggregation = super()._aggregation
            self.update_model = super().update_model

    def _init_params(self):
        self.total_weight = [0]*self.num_clusters
        self.base = [copy.deepcopy(OrderedDict()) for _ in range(self.num_clusters)]

    def is_cluster_key(self, key, cluster_layers="bn"):
        params = {
            "bn": self.batch_norm_names,
            "encoder": self.encoder_names,
            "decoder": self.decoder_names,
            "bn+encoder": self.batch_norm_names + self.encoder_names,
            "bn+decoder": self.batch_norm_names + self.decoder_names,
            "all-stats": [name for name, val in self.model.named_parameters()],
            "all": list(self.model.state_dict().keys())
        }
        return any([p in key for p in params[cluster_layers]])

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):
            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}, Cluster id: {c.cluster_id}")

            if self.clusters_models and "FedAvg" not in self.args.algorithm:
                c.model.load_state_dict(self.clusters_models[c.cluster_id])
            else:
                c.model.load_state_dict(self.model_params_dict)

            if self.args.algorithm == "FedAvg-bn":
                self.load_cluster_batch_norm_stats(c.model, cluster_id=c.cluster_id)

            out = c.train(partial_metric, r=r)

            if self.args.algorithm == "FedAvg-bn":
                self.save_cluster_batch_norm_stats(c.model, cluster_id=c.cluster_id)

            if self.local_rank == 0:
                num_samples, update, dict_losses_list = out
                losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
            else:
                num_samples, update = out

            if self.optimizer is not None:
                raise NotImplementedError

            self.updates.append((num_samples, update, c.cluster_id))

        if self.local_rank == 0:
            return losses

    def _aggregation(self):

        if self.args.algorithm == "FedAvg-bn":
            k_list = []
            total_weight = 0.
            base = OrderedDict()
            for (client_samples, client_model, cluster_id) in self.updates:
                k_list.append(cluster_id)
                cluster_id = int(cluster_id)
                self.total_weight[cluster_id] += client_samples
                total_weight += client_samples
                for key, value in client_model.items():
                    if "running" not in key:
                        if key in base:
                            base[key] += client_samples * value.type(torch.FloatTensor)
                        else:
                            base[key] = client_samples * value.type(torch.FloatTensor)
                    if key in self.base[cluster_id]:
                        self.base[cluster_id][key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        self.base[cluster_id][key] = client_samples * value.type(torch.FloatTensor)

            averaged_sol_n = copy.deepcopy(self.model_params_dict)
            averaged_sol_clusters = [copy.deepcopy(self.model_params_dict) for _ in range(self.num_clusters)]

            for key, value in base.items():
                if total_weight != 0:
                    averaged_sol_n[key] = value.to(self.local_rank) / total_weight

            for key, _ in self.base[k_list[0]].items():
                for cluster_id, base in enumerate(self.base):
                    if self.total_weight[cluster_id] != 0:
                        averaged_sol_clusters[cluster_id][key] = base[key].to(self.local_rank) / self.total_weight[cluster_id]

            return averaged_sol_n, averaged_sol_clusters

        elif self.update_counter <= self.args.fedavg_bootstap:
            total_weight = 0.
            base = OrderedDict()
            for (client_samples, client_model, _) in self.updates:
                total_weight += client_samples
                for key, value in client_model.items():
                    if key in base:
                        base[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base[key] = client_samples * value.type(torch.FloatTensor)
            averaged_sol_n = copy.deepcopy(self.model_params_dict)
            for key, value in base.items():
                if total_weight != 0:
                    averaged_sol_n[key] = value.to(self.local_rank) / total_weight
            return averaged_sol_n

        else:
            k_list = []
            for (client_samples, client_model, cluster_id) in self.updates:
                k_list.append(cluster_id)
                cluster_id = int(cluster_id)
                self.total_weight[cluster_id] += client_samples
                for key, value in client_model.items():
                    if key in self.base[cluster_id]:
                        self.base[cluster_id][key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        self.base[cluster_id][key] = client_samples * value.type(torch.FloatTensor)

            averaged_sol_n = [copy.deepcopy(self.model_params_dict) for _ in range(self.num_clusters)]
            averaged_sol = copy.deepcopy(self.model_params_dict)

            if self.args.algorithm == "PartialClusterAvg":

                for key, _ in self.base[k_list[0]].items():

                    if self.is_cluster_key(key, self.args.cluster_layers):
                        for cluster_id, base in enumerate(self.base):
                            if self.total_weight[cluster_id] != 0:
                                averaged_sol_n[cluster_id][key] = base[key].to(self.local_rank) / self.total_weight[cluster_id]
                    else:
                        total_key_sum = sum([base[key] for base in self.base if key in base.keys()])
                        total_weight_sum = sum(self.total_weight)
                        for cluster_id in range(self.num_clusters):
                            averaged_sol_n[cluster_id][key] = total_key_sum / total_weight_sum
                        averaged_sol[key] = total_key_sum / total_weight_sum

                for key, value in averaged_sol.items():
                    avg = 0
                    if self.is_cluster_key(key, self.args.cluster_layers):
                        for cluster_id in range(self.num_clusters):
                            if self.total_weight[cluster_id] != 0:
                                avg += averaged_sol_n[cluster_id][key]
                        avg /= len(set(k_list))
                        averaged_sol[key] = avg

            elif self.args.algorithm == "TotalClusterAvg":
                for cluster_id in list(set(k_list)):
                    for key, value in self.base[cluster_id].items():
                        if self.total_weight[cluster_id] != 0:
                            averaged_sol_n[cluster_id][key] = value.to(self.local_rank) / self.total_weight[cluster_id]

                for key, value in averaged_sol.items():
                    avg = 0
                    if self.is_cluster_key(key, self.args.cluster_layers) is False:
                        for cluster_id in range(self.num_clusters):
                            if self.total_weight[cluster_id] != 0:
                                avg += averaged_sol_n[cluster_id][key]
                        avg /= len(set(k_list))
                        averaged_sol[key] = avg

                for key, value in averaged_sol.items():
                    if self.is_cluster_key(key, self.args.cluster_layers) is False:
                        for cluster_id in list(set(k_list)):
                            averaged_sol_n[cluster_id][key] = averaged_sol[key]

            return averaged_sol, averaged_sol_n

    def update_model(self):
        if self.update_counter == 0:
            self._init_params()

        self.update_counter += 1

        if self.update_counter <= self.args.fedavg_bootstap:
            self.writer.write("Init Bootstrap")

            averaged_sol_n = self._aggregation()

            if self.optimizer is not None:
                self._server_opt(averaged_sol_n)
                self.total_grad = self._get_model_total_grad()
            else:
                self.model.load_state_dict(averaged_sol_n)
                self.clusters_models = [copy.deepcopy(averaged_sol_n) for _ in range(self.num_clusters)]
            self.model_params_dict = copy.deepcopy(self.model.state_dict())
            self.updates = []

        else:
            if self.args.fedavg_bootstap > 0:
                self.writer.write("END Bootstrap")

            if self.update_counter % self.args.global_aggregation_round == 0:
                self.global_aggregation = True
            else:
                self.global_aggregation = False

            averaged_sol, averaged_sol_n_multi = self._aggregation()
            if self.optimizer is not None:
                raise NotImplementedError
            else:
                if self.args.algorithm == "FedAvg-bn" or self.global_aggregation:
                    self.model.load_state_dict(averaged_sol)
                    self.model_params_dict = copy.deepcopy(self.model.state_dict())
                    self.clusters_models = copy.deepcopy(averaged_sol_n_multi)
                    self._init_params()
                else:
                    self.clusters_models = copy.deepcopy(averaged_sol_n_multi)

            self.updates = []

    def save_cluster_batch_norm_stats(self, model, cluster_id):
        batch_norm_dict = {}
        for name, layer in copy.deepcopy(model).named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                batch_norm_dict[name] = {
                    "running_mean": layer.running_mean,
                    "running_var": layer.running_var
                }
        self.batch_norm_dict[cluster_id] = batch_norm_dict

    def load_cluster_batch_norm_stats(self, model, cluster_id):
        if self.batch_norm_dict[cluster_id] is not None:
            state_dict = model.state_dict()
            for name, layer in copy.deepcopy(model).named_modules():
                if isinstance(layer, nn.BatchNorm2d):
                    state_dict[name + ".running_mean"] = self.batch_norm_dict[cluster_id][name]["running_mean"]
                    state_dict[name + ".running_var"] = self.batch_norm_dict[cluster_id][name]["running_var"]
            model.load_state_dict(state_dict)

    def save_clusters_models(self, r):
        self.writer.write("Saving cluster models, for ablation experiments.")
        for cluster_id in range(self.num_clusters):
            state = {"model_state": self.clusters_models[cluster_id]}
            ckpt_path = os.path.join('checkpoints', self.args.framework, self.args.source_dataset,
                                     self.args.target_dataset, "cluster_models_" + self.writer.wandb.id +
                                     f"{cluster_id}_round_{r}.ckpt")
            torch.save(state, ckpt_path)
            self.writer.wandb.save(ckpt_path)
            self.writer.write(f"saved cluster {cluster_id}")
        self.writer.write("DONE")
