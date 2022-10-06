import torch

from collections import defaultdict
from clients import OracleClient
from utils import Diff2d, Symkl2d


class McdClient(OracleClient):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                         device=device, test_user=test_user)
        self.source_loader = None
        self.discr_loss = self.__get_discr_loss(self.args.discr_loss)

    def __get_discr_loss(self, discr_loss):
        if discr_loss == 'diff2d':
            return Diff2d()
        if discr_loss == 'symkl2d':
            return Symkl2d(size_average=False)
        if self.args.discr_loss == 'symkl2d_size_avg':
            return Symkl2d(size_average=True)
        raise NotImplementedError

    def __unfreeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def __freeze(self, feat_extr=True):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = feat_extr
            elif 'classifier' in name:
                param.requires_grad = not feat_extr

    def calc_loss_and_output(self, source_images, source_labels, target_images=None, phase='1', dict_calc_losses=None):

        if self.args.model == 'deeplabv3':
            outputs_cl1, outputs_cl2 = self.model(source_images, classifier1=True, classifier2=True)

            if phase=='1':
                loss_ph1 = self.reduction(self.criterion(outputs_cl1, source_labels), source_labels) + \
                           self.reduction(self.criterion(outputs_cl2, source_labels), source_labels)
                return {'loss_ph1': loss_ph1}

            if phase == '2':
                loss_ph2 = self.reduction(self.criterion(outputs_cl1, source_labels), source_labels) + \
                           self.reduction(self.criterion(outputs_cl2, source_labels), source_labels) - \
                           self.discr_loss(outputs_cl1, outputs_cl2)
                return {**dict_calc_losses, 'loss_ph2': loss_ph2}

            if phase == '3':
                loss_ph3 = self.discr_loss(outputs_cl1, outputs_cl2) *  self.args.discr_loss_multiplier
                dict_calc_losses['loss_ph3'] = loss_ph3
                return dict_calc_losses

        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer, metric=None, scheduler=None, e_name='EPOCH', plot_lr=True, r=None):

        dict_all_iters_losses = defaultdict(lambda: 0)
        self.loader.sampler.set_epoch(cur_epoch)
        sl_step = r*cur_epoch if r is not None else cur_epoch
        self.source_loader.sampler.set_epoch(sl_step)

        for cur_step, (source_samples, target_samples) in enumerate(zip(self.source_loader, self.loader)):

            source_images = source_samples[0].to(self.device, dtype=torch.float32)
            source_labels = source_samples[1].to(self.device, dtype=torch.long)
            if self.loader.dataset.ds_type == 'unsupervised':
                target_images = target_samples.to(self.device, dtype=torch.float32)
            else:
                target_images = target_samples[0].to(self.device, dtype=torch.float32)

            self.__unfreeze()
            optimizer.zero_grad()
            dict_calc_losses = self.calc_loss_and_output(source_images, source_labels, target_images, phase='1')
            dict_calc_losses['loss_ph1'].backward()
            optimizer.step()

            self.__freeze(feat_extr=False)
            optimizer.zero_grad()
            dict_calc_losses = self.calc_loss_and_output(source_images, source_labels, target_images, phase='2',
                                                         dict_calc_losses=dict_calc_losses)
            dict_calc_losses['loss_ph2'].backward()
            optimizer.step()

            self.__freeze(feat_extr=True)
            for i in range(self.args.repeat_phase_3):
                optimizer.zero_grad()
                dict_calc_losses = self.calc_loss_and_output(source_images, source_labels, target_images, phase='3',
                                                             dict_calc_losses=dict_calc_losses)
                dict_calc_losses['loss_ph3'].backward()
                optimizer.step()

            self.handle_logs(cur_step, cur_epoch, dict_calc_losses, metric, scheduler, plot_lr)

            if scheduler is not None:
                scheduler.step()

            self.update_all_iters_losses(dict_all_iters_losses, dict_calc_losses)

        self.mean_all_iters_losses(dict_all_iters_losses)
        self.writer.write(f"{e_name} {cur_epoch + 1}: ended.")

        return dict_all_iters_losses