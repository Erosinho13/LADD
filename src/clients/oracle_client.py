import copy
import torch

from torch import distributed
from clients.client import Client
from torch.cuda.amp import autocast
from collections import defaultdict
from utils import get_optimizer_and_scheduler


class OracleClient(Client):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                         device=device, test_user=test_user)
        self.teacher_kd_model = None
        self.lambda_kd = -1
        self.server_params = None
        self.server_model = None
        self.mu = self.args.fedprox_mu

    def plot_condition(self, cur_step):
        return (cur_step + 1) % self.args.plot_interval == 0 and self.args.framework == 'centralized'

    def update_metric_condition(self, cur_epoch):
        train_on_server = True if hasattr(self, 'source_client') and self.source_client else False
        return (self.args.framework == 'federated' and cur_epoch == self.args.num_epochs - 1) \
            or (self.args.framework == 'federated' and train_on_server) \
            or self.args.framework == 'centralized'

    def handle_logs(self, cur_step, cur_epoch, dict_calc_losses, metric, scheduler, plot_lr):
        if self.plot_condition(cur_step):
            cum_step = len(self.loader) * cur_epoch + cur_step
            self.writer.plot_step_loss(metric.name, cum_step, dict_calc_losses)
            if plot_lr and scheduler is not None:
                self.writer.plot_step_lr(metric.name, cum_step, scheduler.get_last_lr()[0])
        if (cur_step + 1) % self.args.print_interval == 0:
            self.writer.print_step_loss(str(self), metric.name, cur_step, self.len_loader, dict_calc_losses)

    def update_all_iters_losses(self, dict_all_iters_losses, dict_calc_losses):
        if self.args.framework == 'federated':
            for name, l in dict_calc_losses.items():
                dict_all_iters_losses[name] += l.detach().item() if type(l) != int else l

    def mean_all_iters_losses(self, dict_all_iters_losses):
        if self.args.framework == 'federated':
            for name, l in dict_all_iters_losses.items():
                dict_all_iters_losses[name] /= self.len_loader

    def set_client_teacher(self, cur_step, model):
        if cur_step % self.args.teacher_step == 0:
            self.writer.write(f"step {cur_step}, setting new teacher...")
            if self.args.teacher_kd_step == -1:
                self.writer.write("setting teacher kd too...")
                self.teacher_kd_model = model
            if hasattr(self.criterion, 'set_teacher'):
                self.criterion.set_teacher(model)
            self.writer.write(f"Done.")

    @staticmethod
    def add_4th_layer(images, hpfs):
        tmp = torch.zeros((images.shape[0], images.shape[1] + 1, images.shape[2], images.shape[3]))
        tmp[:, 0:3, :, :] = images
        tmp[:, 3, :, :] = hpfs
        return tmp

    def process_samples(self, loader, samples):
        if loader.dataset.ds_type == 'unsupervised':
            if not self.args.hp_filtered:
                images = samples.to(self.device, dtype=torch.float32)
                labels = None
            else:
                images = self.add_4th_layer(samples[0], samples[1])
                images = images.to(self.device, dtype=torch.float32)
                labels = None
        else:
            images = samples[0].to(self.device, dtype=torch.float32)
            labels = samples[1].to(self.device, dtype=torch.long)
        return images, labels

    def __exec_epoch(self, optimizer, cur_epoch, metric, scheduler, plot_lr, dict_all_iters_losses, profiler=None,
                     stop_at_step=None, r=None):

        self.model.train()

        if self.args.stop_epoch_at_step != -1:
            stop_at_step = self.args.stop_epoch_at_step

        for cur_step, samples in enumerate(self.loader):
            torch.cuda.empty_cache()

            if stop_at_step is not None and cur_step >= stop_at_step:
                break

            if self.args.teacher_step > 0 and self.args.teacher_upd_step and \
                    'fda_inv' in self.args.fw_task and self.args.centr_fda_ft_uda:
                teacher_model = copy.deepcopy(self.model)
                teacher_model.eval()
                self.set_client_teacher(cur_step, teacher_model)

            if self.args.teacher_kd_step > 0 and self.args.teacher_kd_upd_step and \
                    'fda_inv' in self.args.fw_task and self.args.centr_fda_ft_uda:
                if cur_step % self.args.teacher_kd_step == 0 and 'fda_inv' in self.args.fw_task and \
                        self.args.centr_fda_ft_uda:
                    self.writer.write("Setting kd teacher...")
                    teacher_kd_model = copy.deepcopy(self.model)
                    teacher_kd_model.eval()
                    self.teacher_kd_model = teacher_kd_model
                    self.writer.write("Done.")

            images, labels = self.process_samples(self.loader, samples)

            optimizer.zero_grad()

            if self.args.batch_norm_round_0 and r == 0:
                with torch.no_grad():
                    if self.args.mixed_precision:
                        with autocast():
                            dict_calc_losses, outputs = self.calc_loss_and_output(images, labels)
                    else:
                        dict_calc_losses, outputs = self.calc_loss_and_output(images, labels)
                optimizer.zero_grad()
            else:
                if self.args.mixed_precision:
                    with autocast():
                        dict_calc_losses, outputs = self.calc_loss_and_output(images, labels)
                    self.scaler.scale(dict_calc_losses['loss_tot']).backward()
                else:
                    dict_calc_losses, outputs = self.calc_loss_and_output(images, labels)
                    dict_calc_losses['loss_tot'].backward()

            if self.args.fedprox:
                self.handle_grad()
            self.handle_logs(cur_step, cur_epoch, dict_calc_losses, metric, scheduler, plot_lr)
            self.scaler.step(optimizer) if self.args.mixed_precision else optimizer.step()

            if profiler is not None:
                profiler.step()
            if scheduler is not None:
                scheduler.step()

            if self.update_metric_condition(cur_epoch) and not self.args.ignore_train_metrics:
                self.update_metric(metric, outputs, labels)

            if self.args.mixed_precision:
                self.scaler.update()

            self.update_all_iters_losses(dict_all_iters_losses, dict_calc_losses)

    def run_epoch(self, cur_epoch, optimizer, metric=None, scheduler=None, e_name='EPOCH', plot_lr=True,
                  stop_at_step=None, r=None):

        dict_all_iters_losses = defaultdict(lambda: 0)
        self.loader.sampler.set_epoch(cur_epoch)

        if self.profiler_path:
            with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
                                        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_path),
                                        with_stack=True) \
                    as profiler:
                self.__exec_epoch(optimizer, cur_epoch, metric, scheduler, plot_lr, dict_all_iters_losses,
                                  profiler=profiler, stop_at_step=stop_at_step, r=r)
        else:
            self.__exec_epoch(optimizer, cur_epoch, metric, scheduler, plot_lr, dict_all_iters_losses,
                              stop_at_step=stop_at_step, r=r)

        self.mean_all_iters_losses(dict_all_iters_losses)
        self.writer.write(f"{e_name} {cur_epoch + 1}: ended.")

        return dict_all_iters_losses

    def __sync_all_iters_losses(self, dict_losses_list, dict_all_iters_losses):
        for n, l in dict_all_iters_losses.items():
            dict_all_iters_losses[n] = torch.tensor(l).to(self.device)
            distributed.reduce(dict_all_iters_losses[n], dst=0)
            if self.args.local_rank == 0:
                dict_losses_list[n].append(dict_all_iters_losses[n] / distributed.get_world_size())

    def max_iter(self):
        return self.args.num_rounds * self.args.num_epochs * self.len_loader

    def train(self, partial_metric, r=None):

        if self.args.fedprox:
            self.server_model = copy.deepcopy(self.model)
        optimizer, scheduler = get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter())

        dict_losses_list = defaultdict(lambda: [])
        self.model.train()

        for epoch in range(self.args.num_epochs):
            dict_all_iters_losses = self.run_epoch(epoch, optimizer, metric=partial_metric, r=r)
            self.__sync_all_iters_losses(dict_losses_list, dict_all_iters_losses)

        partial_metric.synch(self.device)

        if self.args.fedprox:
            del self.server_model
        if self.args.local_rank == 0:
            return len(self.dataset), copy.deepcopy(self.model.state_dict()), dict_losses_list
        return len(self.dataset), copy.deepcopy(self.model.state_dict())

    def switch_bn_stats_to_test(self):
        for name, layer in self.model.named_modules():
            if 'bn' in name:
                layer.training = True

    def test(self, metric, swa=False):

        self.model.eval()

        if swa:
            self.switch_bn_stats_to_test()

        self.dataset.test = True

        tot_loss = 0.0

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.loader):

                if self.args.stop_epoch_at_step != -1 and i >= self.args.stop_epoch_at_step:
                    break

                if (i + 1) % self.args.print_interval == 0:
                    self.writer.write(f'{self}: {i + 1}/{self.len_loader}, '
                                      f'{round((i + 1) / self.len_loader * 100, 2)}%')

                if self.args.hp_filtered:
                    original_images, images, images_hpf = images
                else:
                    original_images, images = images

                if self.args.hp_filtered:
                    images = self.add_4th_layer(images, images_hpf)

                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self.get_test_output(images)
                self.update_metric(metric, outputs, labels, is_test=True)

                if outputs.shape != labels.shape:
                    outputs = torch.nn.functional.interpolate(outputs, labels.shape[1:], mode='nearest')
                loss = self.calc_test_loss(outputs, labels)
                tot_loss += loss.item()

                torch.cuda.empty_cache()

            metric.synch(self.device)
            mean_loss = self.manage_tot_test_loss(tot_loss)

        self.dataset.test = False

        return {f'{self}_loss': mean_loss}

    def handle_grad(self):
        for client_param, server_param in zip(self.model.parameters(), self.server_model.parameters()):
            client_param.grad.data.add_(client_param.data - server_param.data, alpha=self.mu)
