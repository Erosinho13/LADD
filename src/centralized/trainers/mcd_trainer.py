from metrics import StreamSegMetrics
from .trainer import Trainer


class McdTrainer(Trainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)
        self.target_train_clients[0].source_loader = self.source_train_clients[0].loader

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'train': StreamSegMetrics(num_classes, 'train'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'test_target': StreamSegMetrics(num_classes, 'test_target'),
        }
        writer.write("Done.")
        return metrics

    def handle_ckpt_step(self):
        return None, None, None, self.checkpoint_step

    def max_iter(self):
        return self.args.num_epochs * \
               min(self.target_train_clients[0].len_loader, self.source_train_clients[0].len_loader)

    def train(self, max_scores=None):

        max_scores = [0] * len(self.target_test_clients)

        for e in range(self.ckpt_epoch, self.args.num_epochs):
            self.writer.write(f"EPOCH: {e + 1}/{self.args.num_epochs}")
            self.model.train()
            self.target_train_clients[0].run_epoch(cur_epoch=e, optimizer=self.optimizer, scheduler=self.scheduler,
                                                   metric=self.metrics['train'])
            self.save_model(e + 1, self.optimizer, self.scheduler)
            if (e + 1) % self.args.test_interval == 0 or (e + 1) == self.args.num_epochs:
                max_scores, improvement = self.test(self.target_test_clients, self.metrics['test_target'], e,
                                                    'EPOCH', max_scores, cl_type='target')
                self.test(self.source_test_clients, self.metrics['test_source'], e, 'EPOCH',
                          self.get_fake_max_scores(improvement, len(self.source_test_clients)), cl_type='source')

        return max_scores
