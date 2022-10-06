import torch

from utils import get_scheduler
from metrics import StreamSegMetrics
from centralized.trainers.source_only_trainer import SourceOnlyTrainer


class FtdaTrainer(SourceOnlyTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write('Setting up metrics...')
        metrics = {
            'train_source': StreamSegMetrics(num_classes, 'train_source'),
            'train_target': StreamSegMetrics(num_classes, 'train_target'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'test_target': StreamSegMetrics(num_classes, 'test_target')
        }
        writer.write('Done.')
        return metrics

    def handle_ckpt_step(self):
        if self.checkpoint_step < self.args.num_source_epochs:
            return None, self.checkpoint_step, None, 0
        return None, self.args.num_source_epochs, None, self.checkpoint_step - self.args.num_source_epochs

    def max_iter(self):
        if self.checkpoint_step < self.args.num_source_epochs:
            return self.args.num_source_epochs * self.source_train_clients[0].len_loader
        return self.args.num_epochs * self.target_train_clients[0].len_loader

    def phase1(self):

        max_scores = [0] * len(self.target_test_clients)

        if self.ckpt_source_epoch < self.args.num_source_epochs:
            self.writer.write("\nPHASE 1 - Training on source dataset\n")
            max_scores = SourceOnlyTrainer.train(self, self.metrics['train_source'])
            self.writer.write("\nPHASE 1 completed.\n")

        return max_scores

    def phase2(self, max_scores):

        self.writer.write("\nPHASE 2 - Training on target dataset\n")
        self.checkpoint_step = self.args.num_source_epochs
        self.scheduler = get_scheduler(self.args, self.optimizer, max_iter=self.max_iter())

        for e in range(self.ckpt_epoch, self.args.num_epochs):
            self.perform_centr_oracle_training('EPOCH', e, self.args.num_epochs, self.target_train_clients[0],
                                               self.metrics['train_target'])
            if (e + 1) % self.args.test_interval == 0 or (e + 1) == self.args.num_epochs:
                max_scores, improvement = self.test(self.target_test_clients, self.metrics['test_target'], e, 'EPOCH',
                                                    max_scores, cl_type='target', prepend='phase2_')
                self.test(self.source_test_clients, self.metrics['test_source'], e, 'EPOCH',
                          self.get_fake_max_scores(improvement, len(self.source_test_clients)), cl_type='source',
                          prepend='phase2_')
        self.writer.write("\nPHASE 2 completed.\n")

        return max_scores

    def train(self, train_metric=None, max_scores=None):
        max_scores = self.phase1()
        torch.cuda.empty_cache()
        max_scores = self.phase2(max_scores)
        return max_scores
