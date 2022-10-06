import copy
import numpy as np

from torch import optim


class Server:

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer, source_dataset):
        self.model = copy.deepcopy(model)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.writer = writer
        self.selected_clients = []
        self.updates = []
        self.local_rank = local_rank
        self.opt_string = optimizer
        self.lr = lr
        self.momentum = momentum
        self.optimizer = self.__get_optimizer()
        self.total_grad = 0
        self.source_dataset = source_dataset
        self.swa_model = None

    def train_source(self, *args, **kwargs):
        raise NotImplementedError

    def train_clients(self, *args, **kwargs):
        raise NotImplementedError

    def update_model(self):
        raise NotImplementedError

    def __get_optimizer(self):

        if self.opt_string is None:
            self.writer.write("Running without server optimizer")
            return None

        if self.opt_string == 'SGD':
            return optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.opt_string == 'FedAvgm':
            return optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)

        if self.opt_string == 'Adam':
            return optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

        if self.opt_string == 'AdaGrad':
            return optim.Adagrad(params=self.model.parameters(), lr=self.lr, eps=10 ** (-2))

        raise NotImplementedError

    def select_clients(self, my_round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients
        num_samples = {c.id: c.num_samples for c in clients}
        return num_samples

    @staticmethod
    def num_parameters(params):
        return sum(p.numel() for p in params if p.requires_grad)

    def setup_swa_model(self, swa_ckpt=None):
        self.swa_model = copy.deepcopy(self.model)
        if swa_ckpt is not None:
            self.swa_model.load_state_dict(swa_ckpt)

    def update_swa_model(self, alpha):
        for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
