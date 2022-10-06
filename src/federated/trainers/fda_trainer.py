import os
import torch

from copy import deepcopy
from collections import OrderedDict
from metrics import StreamSegMetrics
from federated.trainers.oracle_trainer import OracleTrainer as OracleTrainerFed
from utils import dynamic_import, get_optimizer_and_scheduler, schedule_cycling_lr
from centralized.trainers.oracle_trainer import OracleTrainer as OracleTrainerCent


class FdaTrainer(OracleTrainerFed, OracleTrainerCent):

    def __init__(self, args, writer, device, rank, world_size):

        super().__init__(args, writer, device, rank, world_size)

        self.teacher_kd_model = None
        self.swa_teacher_model = None

        writer.write(f'Initializing style transfer module in clients...')
        self.set_client_style_tf_obj()
        writer.write(f'Done')

        for clients in (self.source_train_clients, self.source_test_clients):
            for c in clients:
                c.is_source_client()

        self.batch_norm_dict = {"server": None, "client": None}
        self.swa_n = None
        self.swa_teacher_n = None

    def set_client_style_tf_obj(self):
        clients_with_style = [self.target_train_clients]
        if not self.args.style_only_train:
            clients_with_style.append(self.target_test_clients)
        if self.args.source_style_to_source:
            clients_with_style.append(self.source_train_clients)
        for clients in clients_with_style:
            for c in clients:
                c.set_set_style_tf_fn(self.server.styleaug)

    def server_setup(self):
        server_class = dynamic_import(self.args.framework, self.args.fw_task, 'server')
        server = server_class(self.args, self.model, self.writer, self.args.local_rank, self.args.server_lr,
                              self.args.server_momentum, self.args.server_opt,
                              self.source_train_clients[0].dataset)
        return server

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write('Setting up metrics...')
        metrics = {
            'server_train': StreamSegMetrics(num_classes, 'server_train'),
            'uda_train': StreamSegMetrics(num_classes, 'uda_train'),
            'server_test': StreamSegMetrics(num_classes, 'server_test'),
            'target_test': StreamSegMetrics(num_classes, 'target_test'),
            'target_eval': StreamSegMetrics(num_classes, 'target_eval')
        }
        writer.write('Done.')
        return metrics

    def get_optimizer_and_scheduler(self, lr=None):
        return get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter(), lr=lr)

    def max_iter(self):
        return self.args.num_source_epochs * self.source_train_clients[0].len_loader

    def update_swa_teacher_model(self, alpha, model, swa_teacher_model=None):
        swa_teacher_model = self.swa_teacher_model if swa_teacher_model is None else swa_teacher_model
        for param1, param2 in zip(swa_teacher_model.parameters(), model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
        if swa_teacher_model is not None:
            return swa_teacher_model

    def set_client_teacher(self, r, model):

        if r % self.args.teacher_step == 0 and not self.args.teacher_upd_step:

            self.writer.write(f"round {r}, setting new teacher...")

            if self.args.teacher_kd_step == -1 and self.args.lambda_kd > 0:
                self.writer.write(f"Setting kd teacher too...")

            if self.args.swa_teacher_start != -1 and r + 1 > self.args.swa_teacher_start and \
                    ((r - self.args.swa_teacher_start) // self.args.teacher_step) % self.args.swa_teacher_c == 0:
                self.writer.write(f"Number of models: {self.swa_teacher_n}")
                self.update_swa_teacher_model(1.0 / (self.swa_teacher_n + 1), model)
                self.swa_teacher_n += 1

            if self.swa_teacher_model is not None:
                model = self.swa_teacher_model

            for c in self.target_train_clients:
                if self.args.teacher_kd_step == -1:
                    c.teacher_kd_model = model
                if hasattr(c.criterion, 'set_teacher'):
                    if self.args.fw_task == "ladd" and self.server.clusters_models:
                        model.load_state_dict(self.server.clusters_models[c.cluster_id])
                        c.criterion.set_teacher(model)
                    else:
                        c.criterion.set_teacher(model)
                else:
                    break

            if self.args.count_classes_teacher_step != -1:
                if (r // self.args.teacher_step) % self.args.count_classes_teacher_step == 0:
                    self.writer.write("Updating sampling probs...")
                    self.server.count_classes()
                    self.writer.write("Done.")

            self.writer.write(f"Done.")

    def set_client_kd_teacher(self, r, model):
        if r % self.args.teacher_kd_step == 0:
            self.writer.write(f"Setting kd teacher...")
            for c in self.target_train_clients:
                c.teacher_kd_model = model
            self.writer.write(f"Done.")

    def save_pretrained_server(self):
        self.writer.write("Saving server pretrained ckpt...")
        state = {
            "model_state": self.server.model_params_dict,
        }
        ckpt_path = os.path.join('checkpoints', self.args.framework, self.args.source_dataset, self.args.target_dataset,
                                 f'pretrained_server.ckpt')
        torch.save(state, ckpt_path)
        self.writer.wandb.save(ckpt_path)
        print("Done")

    def __unfreeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def __freeze(self, node="server"):
        if node == "server":
            if self.args.freezing == "es_dc":
                self.__freeze_decoder()
            elif self.args.freezing == "ds_ec":
                self.__freeze_encoder()
            elif self.args.freezing == "all_but_one_server":
                self.__freeze_all_but_one_server_(model=self.server.model)
        elif node == "client":
            if self.args.freezing == "es_dc":
                self.__freeze_encoder()
            elif self.args.freezing == "ds_ec":
                self.__freeze_decoder()

    def __freeze_encoder(self):
        for _, param in self.model.module.backbone.named_parameters():
            param.requires_grad = False

    def __freeze_decoder(self):
        for _, param in self.model.module.classifier.named_parameters():
            param.requires_grad = False

    def __freeze_all_but_one_server_(self, model):
        for _, param in model.module.backbone.named_parameters():
            param.requires_grad = False
        for name, param in model.module.classifier.named_parameters():
            if name in ["0.convs.0.0.weight", "0.convs.0.1.weight", "0.convs.0.1.bias"]:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def save_model(self, step, optimizer=None, scheduler=None):

        state = {
            "step": step,
            "model_state": self.server.model_params_dict,
        }

        if self.server.optimizer is not None:
            state["server_optimizer_state"] = self.server.optimizer.state_dict()
        torch.save(state, self.ckpt_path)
        self.writer.wandb.save(self.ckpt_path)

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.server.model_params_dict = deepcopy(self.model.state_dict())
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        if "server_optimizer_state" in self.checkpoint.keys():
            self.server.optimizer.load_state_dict(self.checkpoint["server_optimizer_state"])
            self.writer.write(f"[!] Server optimizer restored.")

    def setup_swa_teacher_model(self, swa_ckpt=None):
        self.swa_teacher_model = deepcopy(self.model)
        if swa_ckpt is not None:
            self.swa_teacher_model.load_state_dict(swa_ckpt)

    def train(self):

        max_scores = [0] * len(self.target_test_clients)

        if self.ckpt_round == 0:

            pretrained_ckpt = None
            if self.args.load_FDA:
                self.writer.write(f'Checking whether pretrained server model exists...')
                pretrained_ckpt = self.writer.wandb.get_fda_pretrained_model()

            if pretrained_ckpt is not None:
                self.writer.write(f'Ckpt found, loading...')
                self.server.model_params_dict = deepcopy(pretrained_ckpt['model_state'])
                self.target_test_clients[0].model.load_state_dict(self.server.model_params_dict)
                self.writer.write(f'Done')
            else:
                self.writer.write('Traning on server data...')
                max_scores = self.server.train_source(train_clients=self.source_train_clients,
                                                      test_clients=self.target_test_clients,
                                                      train_metric=self.metrics['server_train'],
                                                      test_metric=self.metrics['server_test'],
                                                      optimizer=self.optimizer, scheduler=self.scheduler,
                                                      ret_score=self.ret_score,
                                                      device=self.device, test_fn=self.test,
                                                      num_epochs=self.args.num_source_epochs)
                self.save_pretrained_server()
                self.writer.write('Done')
            if self.args.save_samples > 0:
                plot_samples = self.get_plot_samples(self.source_test_clients[0], cl_type="source")
                for plot_sample in plot_samples:
                    self.writer.plot_samples("source", plot_sample, source=True)
            self.server.model.load_state_dict(self.server.model_params_dict)

        if self.args.hp_filtered:

            new_state_dict = OrderedDict()
            for k, v in self.model.state_dict().items():
                name = k[7:]
                new_state_dict[name] = v

            aux = torch.zeros((32, 4, 3, 3))
            aux[:, 0, :, :] = deepcopy(new_state_dict['backbone.0.0.weight'][:, 0, :, :])
            aux[:, 1, :, :] = deepcopy(new_state_dict['backbone.0.0.weight'][:, 1, :, :])
            aux[:, 2, :, :] = deepcopy(new_state_dict['backbone.0.0.weight'][:, 2, :, :])
            aux[:, 3, :, :] = deepcopy(new_state_dict['backbone.0.0.weight'][:, 2, :, :])
            new_state_dict['backbone.0.0.weight'] = aux

            self.centr_model.load_state_dict(new_state_dict)
            self.model = self.model_init(self.args, self.device, model=self.centr_model)
            self.server.model = self.model
            self.server.model_params_dict = deepcopy(self.model.state_dict())
            for c in [*self.target_train_clients, *self.target_test_clients]:
                c.model = self.model

        if self.args.pretrain:
            return max_scores
        max_scores = [0] * len(self.target_test_clients)

        for r in range(self.ckpt_round, self.args.num_rounds):

            torch.cuda.empty_cache()

            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')
            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)

            if self.args.swa_start != -1 and r + 1 >= self.args.swa_start:
                if r + 1 == self.args.swa_start:
                    self.writer.write("Setting up SWA...")
                    self.server.setup_swa_model()
                if self.args.swa_c > 1:
                    lr = schedule_cycling_lr(r, self.args.swa_c, self.args.lr_fed, self.args.swa_lr)
                    self.server.update_clients_lr(lr)

            if self.args.swa_teacher_start != -1 and r + 1 >= self.args.swa_teacher_start:
                if r + 1 == self.args.swa_teacher_start:
                    self.writer.write("Setting up SWA teacher...")
                    self.setup_swa_teacher_model()

            if self.args.train_source_round_interval is not None and \
                    (r + 1) % self.args.train_source_round_interval == 0:
                self.writer.write('Traning on server data...')
                if (r + 1) // self.args.train_source_round_interval == 1:
                    self.args.num_source_epochs = round(
                        self.args.num_source_epochs * self.args.num_source_epochs_factor_retrain)

                if self.args.freezing is not None:
                    self.__unfreeze()
                    self.__freeze(node="server")

                if self.args.only_update_bn_server is False:
                    self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(
                        lr=self.args.lr * self.args.lr_factor_server_retrain)
                    _ = self.server.train_source(train_clients=self.source_train_clients,
                                                 test_clients=self.target_test_clients,
                                                 train_metric=self.metrics['server_train'],
                                                 test_metric=self.metrics['server_test'],
                                                 optimizer=self.optimizer, scheduler=self.scheduler,
                                                 ret_score=self.ret_score,
                                                 device=self.device, test_fn=self.test,
                                                 num_epochs=self.args.num_source_epochs,
                                                 num_steps=self.args.num_source_steps_retrain)
                    self.writer.write('Done')
                else:
                    self.writer.write('Skipping server train')

            if self.args.freezing is not None:
                self.__unfreeze()
                self.__freeze(node="client")
            if self.args.teacher_step > 0 and not self.args.teacher_upd_step:
                self.model.load_state_dict(self.server.model_params_dict)
                teacher_model = deepcopy(self.model)
                teacher_model.eval()
                self.set_client_teacher(r, teacher_model)
            if self.args.teacher_kd_step > 0 and not self.args.teacher_kd_upd_step:
                teacher_kd_model = deepcopy(self.model)
                teacher_kd_model.eval()
                self.set_client_kd_teacher(r, teacher_kd_model)
            if self.args.teacher_kd_step > 0 and self.args.teacher_kd_mult_factor != -1 and \
                    'fda_inv' in self.args.fw_task and r % self.args.teacher_kd_mult_step == 0 and r != 0:
                self.writer.write(f"Updating lambda_kd={self.target_train_clients[0].lambda_kd} at epoch {r}...")
                self.target_train_clients[0].lambda_kd *= self.args.teacher_kd_mult_factor
                self.writer.write(f"Done. New lambda_kd={self.target_train_clients[0].lambda_kd}")

            losses = self.server.train_clients(partial_metric=self.metrics['uda_train'], r=r)

            self.plot_train_metric(r, self.metrics['uda_train'], losses, plot_metric=False)
            self.metrics['uda_train'].reset()

            if self.args.centr_fda_ft_uda:
                self.server.model_params_dict = deepcopy(self.model.state_dict())
            else:
                self.server.update_model()
                self.model.load_state_dict(self.server.model_params_dict)
                if self.args.algorithm != "FedAvg":
                    self.target_test_clients[0].clusters_models = self.server.clusters_models

            self.save_model(r + 1, optimizer=self.server.optimizer)

            if self.args.swa_start != -1 and r + 1 > self.args.swa_start and \
                    (r - self.args.swa_start) % self.args.swa_c == 0:
                self.writer.write(f"Number of models: {self.swa_n}")
                self.server.update_swa_model(1.0 / (self.swa_n + 1))
                self.swa_n += 1

            if (r + 1) % self.args.eval_interval == 0 and self.all_target_client.loader.dataset.ds_type not in (
                    'unsupervised',):
                self.test([self.all_target_client], self.metrics['target_eval'], r, 'ROUND',
                          self.get_fake_max_scores(False, 1), cl_type='target')

            if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                max_scores, _ = self.test(self.target_test_clients, self.metrics['target_test'], r, 'ROUND', max_scores,
                                          cl_type='target')
                if self.args.test_source:
                    self.test(self.source_test_clients, self.metrics['server_test'], r, 'ROUND',
                              self.get_fake_max_scores(False, 1), cl_type='source')

            if self.args.save_cluster_models and self.args.fw_task == "ladd" and (r + 1) % 100 == 0:
                self.server.save_clusters_models(r + 1)

        return max_scores
