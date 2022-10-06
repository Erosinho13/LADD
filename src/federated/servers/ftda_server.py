from .oracle_server import OracleServer


class FtdaServer(OracleServer):

    def __init__(self, model, writer, local_rank, lr, momentum, optimizer, source_dataset):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer, source_dataset=source_dataset)
