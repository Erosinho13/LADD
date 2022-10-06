import copy
import torch
from collections import OrderedDict
from federated.servers.server import Server


class OracleServer(Server):

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer=optimizer, source_dataset=source_dataset)

    def train_source(self, *args, **kwargs):
        pass

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):

            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")

            c.model.load_state_dict(self.model_params_dict)
            out = c.train(partial_metric, r=r)

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

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, client_model) in self.updates:
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

    def _server_opt(self, pseudo_gradient):
        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.optimizer.step()
        bn_layers = \
            OrderedDict({k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        self.writer.write(f"total grad norm: {round(total_grad, 2)}")
        return total_grad

    def update_model(self):

        averaged_sol_n = self._aggregation()

        if self.optimizer is not None:
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []
