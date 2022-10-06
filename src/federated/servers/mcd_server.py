from .oracle_server import OracleServer


class McdServer(OracleServer):

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer=optimizer, source_dataset=source_dataset)
