from federated.trainers import FdaTrainer, OracleTrainer


class FdaInvTrainer(FdaTrainer):

    def __init__(self, args, writer, device, rank, world_size):
        super(OracleTrainer, self).__init__(args, writer, device, rank, world_size)
        self.swa_teacher_model = None

        for clients in (self.source_train_clients, self.source_test_clients):
            for c in clients:
                c.is_source_client()

        if not args.load_FDA:
            writer.write(f'LADD Initializing style transfer module in clients...')
            self.set_server_style()
            writer.write(f'Done')

        if self.args.swa_start != -1:
            self.swa_n = 0

        if self.args.swa_teacher_start != -1:
            self.swa_teacher_n = 0

    def set_server_style(self):
        for client in self.target_train_clients:
            self.server.styleaug.add_style(client.dataset, multiple_styles=self.args.multiple_styles)

        self.source_train_clients[0].loader.dataset.set_style_tf_fn(self.server.styleaug.apply_style)

    def set_client_style_tf_obj(self):
        pass
