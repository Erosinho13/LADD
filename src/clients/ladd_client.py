from clients import FdaInvClient
import torch
from utils import StyleAugment
from tqdm import tqdm


class LADDClient(FdaInvClient):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank,
                         num_gpu, device=device, test_user=test_user)

        self.cluster_id = None
        self.clusters_models = []
        self.styleaug_test = None
        self.test_images_clusters = []

        if args.test_only_with_global_model:
            self.test = super().test

    def find_test_images_cluster(self, k_means_model):
        self.dataset.return_unprocessed_image = True
        self.styleaug_test = StyleAugment(1, self.args.fda_L, self.args.fda_size, b=self.args.fda_b)

        for sample in tqdm(self.dataset):
            img_processed = self.styleaug_test.preprocess(sample)
            style = self.styleaug_test._extract_style(img_processed)
            cluster = k_means_model.predict(style.reshape(1, -1))[0]
            self.test_images_clusters.append(cluster)

        self.dataset.return_unprocessed_image = False

    def test(self, metric, swa=False):

        tot_loss = 0.0

        self.model.eval()

        if swa:
            self.switch_bn_stats_to_test()

        self.dataset.test = True

        global_model_dict = self.model.state_dict()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.loader):

                if (i + 1) % self.args.print_interval == 0:
                    self.writer.write(f'{self}: {i + 1}/{self.len_loader}, '
                                      f'{round((i + 1) / self.len_loader * 100, 2)}%')

                original_images, images = images
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                self.model.load_state_dict(global_model_dict)
                outputs_global = self.get_test_output(images)
                self.update_metric(metric[0], outputs_global, labels, is_test=True)

                if self.args.algorithm != "FedAvg" and self.clusters_models:
                    self.model.load_state_dict(self.clusters_models[self.test_images_clusters[i]])
                    outputs = self.get_test_output(images)
                    self.update_metric(metric[1], outputs, labels, is_test=True)

                if outputs_global.shape != labels.shape:
                    outputs_global = torch.nn.functional.interpolate(outputs_global, labels.shape[1:], mode='nearest')
                loss = self.calc_test_loss(outputs_global, labels)
                tot_loss += loss.item()

                torch.cuda.empty_cache()

            metric[0].synch(self.device)
            metric[1].synch(self.device)

            mean_loss = self.manage_tot_test_loss(tot_loss)

        self.dataset.test = False

        return {f'{self}_loss': mean_loss}
