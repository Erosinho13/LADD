from .fda_server import FdaServer
from utils import StyleAugment
from .oracle_server import OracleServer


class FdaInvServer(FdaServer):

    def __init__(self, args, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super(OracleServer, self).__init__(model, writer, local_rank, lr, momentum, optimizer,
                                           source_dataset=source_dataset)

        self.args = args
        self.styleaug = StyleAugment(args.n_images_per_style, args.fda_L, args.fda_size, b=args.fda_b)
        self.batch_norm_dict = {"server": None, "client": []}
        self.swa_model = None
