from federated.trainers.trainer import Trainer
from metrics import StreamSegMetrics


class McdTrainer(Trainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)
        for i in range(len(self.target_train_clients)):
            self.target_train_clients[i].source_loader = self.source_train_clients[0].loader

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'partial_train': StreamSegMetrics(num_classes, 'partial_train'),
            'test_source': StreamSegMetrics(num_classes, 'test_source'),
            'test_target': StreamSegMetrics(num_classes, 'test_target'),
        }
        writer.write("Done.")
        return metrics

    def handle_ckpt_step(self):
        return None, None, self.checkpoint_step, None

    def train(self):

        max_scores = [0] * len(self.target_test_clients)

        for r in range(self.ckpt_round, self.args.num_rounds):

            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')
            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)
            losses = self.server.train_clients(r=r, partial_metric=self.metrics['partial_train'])
            self.plot_train_metric(r, self.metrics['partial_train'], losses)
            self.metrics['partial_train'].reset()

            self.server.update_model()
            self.model.load_state_dict(self.server.model_params_dict)
            self.save_model(r + 1, optimizer=self.server.optimizer)

            if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                self.test(self.source_test_clients, self.metrics['test_source'], r, 'ROUND',
                          self.get_fake_max_scores(False, 1), cl_type='source')
                max_scores, _ = self.test(self.target_test_clients, self.metrics['test_target'], r, 'ROUND', max_scores,
                                          cl_type='target')

        return max_scores
