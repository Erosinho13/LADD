import copy

from federated.trainers.trainer import Trainer
from centralized import FtdaTrainer as CentrFtdaTrainer
from metrics import StreamSegMetrics


class FtdaNoaggrTrainer(Trainer, CentrFtdaTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    def set_metrics(self, writer, num_classes):

        writer.write('Setting up metrics...')

        metrics = {
            'train_source': StreamSegMetrics(num_classes, 'train_source'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'test_target': StreamSegMetrics(num_classes, 'test_target'),
        }

        for cl in self.target_train_clients:
            metrics[f'train_{cl}'] = StreamSegMetrics(num_classes, f'train_{cl}')
            metrics[f'test_{cl}'] = StreamSegMetrics(num_classes, f'test_{cl}')

        writer.write('Done.')

        return metrics

    def handle_ckpt_step(self):
        return CentrFtdaTrainer.handle_ckpt_step(self)

    def phase1(self):
        self.optimizer, self.scheduler = CentrFtdaTrainer.get_optimizer_and_scheduler(self)
        return super().phase1()

    def phase2(self, max_scores):
        self.writer.write("\nPHASE 2 - Training on target datasets (clients)\n")
        self.server.model = copy.deepcopy(self.model)
        self.server.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.server.select_clients(1, self.target_train_clients, num_clients=len(self.target_train_clients))
        mean_score = self.server.train_clients(metrics=self.metrics, target_test_client=self.target_test_clients[0],
                                               test_interval=self.args.test_interval, ret_score=self.ret_score)
        self.writer.write("\nPHASE 2 completed.\n")
        return mean_score

    def train(self):
        return CentrFtdaTrainer.train(self)
