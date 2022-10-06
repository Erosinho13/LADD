from metrics import StreamSegMetrics
from federated.trainers.trainer import Trainer


class OracleTrainer(Trainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'test': StreamSegMetrics(num_classes, 'test'),
            'partial_train': StreamSegMetrics(num_classes, 'partial_train'),
            'eval_train': StreamSegMetrics(num_classes, 'eval_train')
        }
        writer.write("Done.")
        return metrics

    def handle_ckpt_step(self):
        return None, None, self.checkpoint_step, None

    def perform_fed_oracle_training(self, partial_train_metric, eval_train_metric, test_metric, max_scores=None):

        if max_scores is None:
            max_scores = [0] * len(self.target_test_clients)

        for r in range(self.ckpt_round, self.args.num_rounds):

            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')
            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)
            losses = self.server.train_clients(partial_metric=partial_train_metric)
            self.plot_train_metric(r, partial_train_metric, losses)
            partial_train_metric.reset()

            self.server.update_model()
            self.model.load_state_dict(self.server.model_params_dict)
            self.save_model(r + 1, optimizer=self.server.optimizer)

            if (r + 1) % self.args.eval_interval == 0 and \
                    self.all_target_client.loader.dataset.ds_type not in ('unsupervised',):
                self.test([self.all_target_client], eval_train_metric, r, 'ROUND', self.get_fake_max_scores(False, 1),
                          cl_type='target')
            if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                max_scores, _ = self.test(self.target_test_clients, test_metric, r, 'ROUND', max_scores,
                                          cl_type='target')

        return max_scores

    def train(self):
        return self.perform_fed_oracle_training(
            partial_train_metric=self.metrics['partial_train'],
            eval_train_metric=self.metrics['eval_train'],
            test_metric=self.metrics['test']
        )
