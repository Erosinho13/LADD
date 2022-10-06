from .ftda_noaggr_trainer import FtdaNoaggrTrainer
from .oracle_trainer import OracleTrainer
from metrics import StreamSegMetrics


class FtdaTrainer(FtdaNoaggrTrainer, OracleTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    def set_metrics(self, writer, num_classes):
        writer.write('Setting up metrics...')
        metrics = {
            'train_source': StreamSegMetrics(num_classes, 'train_source'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'partial_train_target': StreamSegMetrics(num_classes, 'partial_train_target'),
            'eval_train_target': StreamSegMetrics(num_classes, 'eval_train_target'),
            'test_target': StreamSegMetrics(num_classes, 'test_target')
        }
        writer.write('Done.')
        return metrics

    def handle_ckpt_step(self):
        if self.checkpoint_step < self.args.num_source_epochs:
            return None, self.checkpoint_step, 0, 0
        return None, self.args.num_source_epochs, self.checkpoint_step - self.args.num_source_epochs, 0

    def phase2(self, max_scores):
        self.writer.write("\nPHASE 2 - Training on target datasets (clients) and aggregating\n")
        max_scores = super().perform_fed_oracle_training(
            partial_train_metric=self.metrics['partial_train_target'],
            eval_train_metric=self.metrics['eval_train_target'],
            test_metric=self.metrics['test_target'],
            max_scores=max_scores
        )
        self.writer.write("\nPHASE 2 completed.\n")
        return max_scores

    def train(self):
        return FtdaNoaggrTrainer.train(self)
