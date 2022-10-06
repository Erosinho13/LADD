import copy
import math
import numpy as np
import torch
from torch import nn
from collections import defaultdict

from tqdm import tqdm

from clients import OracleClient
from utils import HardNegativeMining, MeanReduction, AdvEntLoss, IW_MaxSquareloss, SelfTrainingLoss, \
                  SelfTrainingLossEntropy, SelfTrainingLossLovaszEntropy, EntropyLoss, LovaszLoss, \
                  get_optimizer_and_scheduler, KnowledgeDistillationLoss


class FdaClient(OracleClient):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                         device=device, test_user=test_user)

        self.styleaug = None
        self.source_client = False
        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()
        self.test_criterion, self.test_reduction = self.__get_criterion_and_reduction_rules(use_labels=True)
        self.entropy_loss = EntropyLoss(lambda_entropy=self.args.lambda_entropy, num_classes=self.args.num_classes)
        self.kd_loss = KnowledgeDistillationLoss(reduction='mean', alpha=self.args.alpha_kd)
        self.teacher_kd_model = None
        self.lambda_kd = self.args.lambda_kd
        self.class_probs = None

    def set_set_style_tf_fn(self, styleaug):
        self.styleaug = styleaug
        self.loader.dataset.set_style_tf_fn(self.styleaug.apply_style)

    def is_source_client(self):
        self.source_client = True
        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()

    def calc_loss_and_output(self, images, labels):

        if self.source_client:
            return super().calc_loss_and_output(images, labels)

        def pseudo(outs):
            return outs.max(1)[1]

        kwargs = {}
        if self.args.teacher_step > 0:
            kwargs['imgs'] = images
            if self.args.count_classes_teacher_step != -1 and self.args.weights_lovasz:
                kwargs['weights'] = self.class_probs

        if self.args.model in ('deeplabv3',):

            if "div" in self.args.fda_loss:

                outputs = self.model(images)['out']
                with torch.no_grad():
                    outputs_old = self.teacher_kd_model(images)['out'] if self.args.lambda_kd > 0 else None

                self_loss = self.reduction(self.criterion(outputs, **kwargs), pseudo(outputs))
                entropy_loss = self.entropy_loss(outputs)

                dict_calc_losses = {'self_loss': self_loss, 'entropy_loss': entropy_loss}

                if outputs_old is not None:
                    pseudo_labels = self.criterion.get_pseudo_lab(outputs_old, images, model=self.teacher_kd_model)
                    mask = torch.ones(pseudo_labels.shape).double().to(self.device)
                    mask = torch.where(pseudo_labels != 255, mask, 0.) if pseudo_labels is not None else None
                    kd_loss = self.kd_loss(outputs, outputs_old, pred_labels=labels, mask=mask)
                    dict_calc_losses = {**dict_calc_losses, 'kd_loss': kd_loss}
                    loss_tot = self_loss + entropy_loss + self.lambda_kd * kd_loss
                else:
                    loss_tot = self_loss + entropy_loss

                dict_calc_losses = {**dict_calc_losses, 'loss_tot': loss_tot}

            else:
                outputs = self.model(images)['out']
                loss_tot = self.reduction(self.criterion(outputs, **kwargs), pseudo(outputs))
                dict_calc_losses = {'loss_tot': loss_tot}

        else:
            raise NotImplementedError

        return dict_calc_losses, outputs

    def calc_test_loss(self, outputs, labels):
        if self.args.model in ('deeplabv3',) and "div" in self.args.fda_loss:
            lovasz_loss = self.test_reduction(self.test_criterion(outputs, labels), labels)
            entropy_loss = self.entropy_loss(outputs)
            return lovasz_loss + entropy_loss
        else:
            return self.test_reduction(self.test_criterion(outputs, labels), labels)

    def plot_condition(self, cur_step):
        return (cur_step + 1) % self.args.plot_interval == 0

    def update_metric(self, metrics, outputs, labels, is_test=False):
        if not self.source_client and not is_test:
            return
        _, prediction = outputs.max(dim=1)
        if prediction.shape != labels.shape:
            prediction = nn.functional.interpolate(
                prediction.unsqueeze(0).double(), labels.shape[1:], mode='nearest').squeeze(0).long()
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metrics.update(labels, prediction)

    def count_classes(self):
        class_freqs = defaultdict(lambda: 0)
        class_by_image = \
            defaultdict(lambda: [])
        self.model.eval()
        self.dataset.test = True
        for i, sample in enumerate(tqdm(self.loader_full, maxinterval=len(self.dataset))):
            image, _ = self.process_samples(self.loader_full, sample[0][0])
            with torch.no_grad():
                output = self.model(image)['out']
                pseudo = self.criterion.get_pseudo_lab(output, image)
                np_pseudo = pseudo.cpu().detach().numpy()
                unique, counts = np.unique(np_pseudo, return_counts=True)
                for cl, count in zip(unique, counts):
                    if cl == 255:
                        continue
                    class_freqs[cl] += count
                    class_by_image[cl].append(i)
        class_freqs = {k: v / sum(class_freqs.values()) * 100 for k, v in class_freqs.items()}
        class_probs = {k: math.exp(1-class_freqs[k])/self.args.temperature for k in class_freqs.keys()}
        class_probs = {k: v / sum(class_probs.values()) for k, v in class_probs.items()}
        self.class_probs = class_probs
        labels = ["road", "sidewalk", "building", "traffic light", "traffic sign", "vegetation", "sky",
                  "person", "rider", "car", "bus", "motorcycle", "bicycle"]
        cprob_to_print = {labels[k]: round(v, 3) for k, v in class_probs.items()}
        self.writer.write(f"Extracted class probs for client {self}: {cprob_to_print}")
        self.dataset.test = False
        self.model.train()

        return class_probs, class_by_image

    def train(self, partial_metric, r=None):

        if self.args.fedprox:
            self.server_model = copy.deepcopy(self.model)
        optimizer, _ = get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter(),
                                                   lr=self.args.lr_fed if not self.source_client else None)

        dict_losses_list = defaultdict(lambda: [])
        self.model.train()

        if self.args.disable_batch_norm:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    m.track_running_stats = False

        for epoch in range(self.args.num_epochs):
            dict_all_iters_losses = self.run_epoch(epoch, optimizer, metric=partial_metric, r=r)
            self._OracleClient__sync_all_iters_losses(dict_losses_list, dict_all_iters_losses)

        partial_metric.synch(self.device)

        if self.args.fedprox:
            del self.server_model
        if self.args.local_rank == 0:
            return len(self.dataset), copy.deepcopy(self.model.state_dict()), dict_losses_list
        return len(self.dataset), copy.deepcopy(self.model.state_dict())

    def __get_criterion_and_reduction_rules(self, use_labels=False):

        loss_choices = {'advent': AdvEntLoss,
                        'maxsquares': IW_MaxSquareloss,
                        'selftrain': SelfTrainingLoss,
                        'selftrainentropy': SelfTrainingLossEntropy,
                        'lovasz_entropy_joint': SelfTrainingLossLovaszEntropy,
                        'lovasz_entropy_div': LovaszLoss,
                        'selftrain_div': SelfTrainingLoss}
        loss_fn = nn.CrossEntropyLoss if (use_labels or self.source_client) else loss_choices[self.args.fda_loss]

        shared_kwargs = {'ignore_index': 255, 'reduction': 'none'}
        if not (use_labels or self.source_client):
            if self.args.fda_loss == "lovasz_entropy_div":
                criterion = loss_fn = LovaszLoss(lambda_selftrain=self.args.lambda_selftrain, **shared_kwargs)
            elif self.args.fda_loss == "selftrain_div":
                criterion = loss_fn = SelfTrainingLoss(lambda_selftrain=self.args.lambda_selftrain, **shared_kwargs)
        else:
            criterion = loss_fn(**shared_kwargs)
        if hasattr(loss_fn, 'requires_reduction') and not loss_fn.requires_reduction:
            reduction = lambda x, y: x
        else:
            reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        return criterion, reduction
