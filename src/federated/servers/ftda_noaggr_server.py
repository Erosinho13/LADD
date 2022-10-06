from .oracle_server import OracleServer


class FtdaNoaggrServer(OracleServer):

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer, source_dataset):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer, source_dataset=source_dataset)

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):
        score = 0
        clients = self.selected_clients
        for i, c in enumerate(clients):
            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")
            c.model.load_state_dict(self.model_params_dict)
            c.train_target(metrics=metrics, target_test_client=target_test_client, test_interval=test_interval,
                           ret_score=ret_score)
            score += metrics[f'test_{c}'].get_results()[ret_score]
        return score/len(clients)
