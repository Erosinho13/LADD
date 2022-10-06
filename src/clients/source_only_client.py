from clients import OracleClient


class SourceOnlyClient(OracleClient):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                         device=device, test_user=test_user)
