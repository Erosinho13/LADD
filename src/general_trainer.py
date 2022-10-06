import copy
import os
import cv2
import torch
import matplotlib
import numpy as np
import seaborn as sns
import yaml

from torch import nn
from modules import McdWrapper
from collections import OrderedDict
from utils import make_model, DatasetHandler, dynamic_import
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')


class GeneralTrainer(object):

    def __init__(self, args, writer, device, rank, world_size):

        self.args = args
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.centr_model = None

        if self.args.load_yaml_config is not None:
            writer.write(f'Update args from yaml config provided')
            with open(self.args.load_yaml_config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self.args = {**vars(self.args), **{key: value for key, value in yaml_config.items()
                                                   if key in vars(self.args) and key != 'wandb_id'}}
        writer.write(f'Initializing model...')
        self.model = self.model_init(args, device)
        writer.write('Done.')

        writer.write(f'Initializing datasets...')
        self.clients_args = DatasetHandler(args, writer)()
        writer.write('Done.')

        writer.write('Initializing clients...')
        self.clients_shared_args = {'args': self.args, 'model': self.model,
                                    'writer': self.writer, 'world_size': world_size, 'rank': rank,
                                    'num_gpu': args.n_devices, 'device': device}
        self.source_train_clients, self.source_test_clients = [], []
        self.target_train_clients, self.target_test_clients = [], []
        self.__clients_setup()
        writer.write('Done.')

        writer.write(f'Initializing server...')
        self.server = self.server_setup()
        writer.write('Done.')

        writer.write('Initialize return score, metrics, ckpt, ckpt step...')
        self.ret_score = self.__get_ret_score()
        self.metrics = self.set_metrics(writer, args.num_classes)
        self.checkpoint_step = 0
        self.ckpt_path = os.path.join('checkpoints', args.framework, args.source_dataset, args.target_dataset,
                                      f"{self.writer.wandb.get_job_name(args)}_{self.args.wandb_id}.ckpt")
        self.checkpoint, self.checkpoint_step = self.__preload()
        self.ckpt_source_round, self.ckpt_source_epoch, self.ckpt_round, self.ckpt_epoch = self.handle_ckpt_step()
        writer.write('Done.')

        writer.write('Initializing optimizer and scheduler...')
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()
        writer.write('Done.')

        if rank == 0:
            if not os.path.exists(os.path.dirname(self.ckpt_path)):
                os.makedirs(os.path.dirname(self.ckpt_path))

        if args.load:
            writer.write("Loading model from checkpoint...")
            self.__load_wandb()
            self.load_from_checkpoint()
            writer.write("Done.")

        self.sample_ids = None
        if self.args.save_samples > 0:
            writer.write("Generating sample ids for plots...")
            self.__gen_all_sample_ids()
            self.test_plot_counter = 0
            writer.write("Done.")
        else:
            self.test_plot_counter = -1

        self.train_args = self.get_train_args()
        self.train_kwargs = self.get_train_kwargs()

    def model_init(self, args, device, model=None):

        if model is None:
            model = make_model(args)
            if args.fw_task == 'mcd':
                model = McdWrapper(model=model)

        model = model.to(device)

        if args.hp_filtered:
            self.centr_model = make_model(args, augm_model=True)
            self.centr_model.to(device)

        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                    find_unused_parameters=args.fw_task == 'mcd'
                                                    or ('fda' in args.fw_task and (args.disable_batch_norm or
                                                                                   args.freezing is not None)))

        return model

    def server_setup(self):
        raise NotImplementedError

    def __clients_setup(self):
        client_class = dynamic_import(self.args.framework, self.args.fw_task, 'client')
        for split, cl_data_args in self.clients_args.items():
            if split == 'all_train':
                continue
            for cl_data_arg in cl_data_args:
                batch_size = self.args.batch_size if split == 'train' else self.args.test_batch_size
                cl_args = {**self.clients_shared_args, **cl_data_arg}
                cl = client_class(**cl_args, batch_size=batch_size, test_user=split == 'test')
                if 'source' not in str(cl):
                    self.target_train_clients.append(cl) if split == 'train' else self.target_test_clients.append(cl)
                else:
                    self.source_train_clients.append(cl) if split == 'train' else self.source_test_clients.append(cl)

    def __get_ret_score(self):
        if self.model.module.task == 'classification':
            return 'Overall Acc'
        if self.model.module.task == 'segmentation':
            return 'Mean IoU'
        raise NotImplementedError

    @staticmethod
    def set_metrics(writer, num_classes):
        raise NotImplementedError

    def __preload(self):
        if self.args.load:
            checkpoint = torch.load(self.ckpt_path)
            return checkpoint, checkpoint['step']
        return None, 0

    def handle_ckpt_step(self):
        raise NotImplementedError

    def get_optimizer_and_scheduler(self):
        raise NotImplementedError

    def __load_wandb(self):
        run_path = os.path.join(
            self.args.wandb_entity,
            self.writer.wandb.get_project_name(self.args.framework, self.args.source_dataset, self.args.target_dataset),
            self.args.wandb_id
        )
        self.writer.wandb.restore(name=self.ckpt_path, run_path=run_path, root=".")

    def load_from_checkpoint(self):
        raise NotImplementedError

    def __gen_sample_ids(self, clients, cl_type='target'):
        self.sample_ids[cl_type] = {}
        for test_cl in clients:
            self.sample_ids[cl_type][str(test_cl)] = \
                np.random.choice(len(test_cl.dataset), self.args.save_samples, replace=False)

    def __gen_all_sample_ids(self):
        self.sample_ids = {'source': [], 'target': []}
        self.__gen_sample_ids(self.source_test_clients, cl_type='source')
        self.__gen_sample_ids(self.target_test_clients, cl_type='target')

    @staticmethod
    def get_train_args():
        return []

    @staticmethod
    def get_train_kwargs():
        return {}

    def save_model(self, step, optimizer=None, scheduler=None):
        raise NotImplementedError

    def __get_plot_sample(self, test_client, sample_id):

        plot_sample = {}

        sample = test_client.dataset[sample_id]

        self.model.eval()

        with torch.no_grad():
            sample_pred = test_client.get_test_output(sample[0][0].unsqueeze(0)).argmax(dim=1)[0].detach().cpu().numpy()
            sample_img = sample[0][0].detach().numpy()
            if sample[0][0].shape != sample[1].shape:
                sample_label = nn.functional.interpolate(
                    sample[1].unsqueeze(0).unsqueeze(0).double(),
                    sample[0][0].shape[1:], mode='nearest').squeeze(0).squeeze(0).long()
            sample_label = sample_label.detach().numpy()
        plot_sample[str(test_client)] = OrderedDict(
            {'sid': sample_id, 'img': sample_img, 'label': sample_label, 'pred': sample_pred})

        if hasattr(test_client.criterion, 'get_pseudo_lab'):
            with torch.no_grad():
                pred_torch = test_client.get_test_output(sample[0][0].unsqueeze(0))
                pseudo_lab, softmax, mask_fract = test_client.criterion.get_pseudo_lab(pred=pred_torch,
                                                                                       imgs=sample[0][0].unsqueeze(0),
                                                                                       return_mask_fract=True)
            plot_sample[str(test_client)][f'Pseudo Label ({mask_fract * 100:.2f}%)'] = pseudo_lab[
                0].detach().cpu().numpy()

            fig = Figure(figsize=(16, 8), dpi=128)
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            sns.heatmap(softmax.detach().cpu().numpy().max(1)[0], ax=ax)
            fig.canvas.draw()
            heatmap_np = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            heatmap_np = heatmap_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            heatmap_np = cv2.resize(heatmap_np, dsize=sample_img.shape[1:3][::-1], interpolation=cv2.INTER_CUBIC)
            plot_sample[str(test_client)]['Pseudo Confidence'] = heatmap_np.transpose((2, 0, 1))

        return plot_sample

    def get_plot_samples(self, test_client, cl_type='target'):

        plot_samples = []

        test_client.dataset.test = True

        for i in self.sample_ids[cl_type][str(test_client)]:
            plot_samples.append(self.__get_plot_sample(test_client, i))

        test_client.dataset.test = False

        return plot_samples

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def perform_test(self, metric, test_clients, step):
        self.writer.write("Testing...")
        scores = []
        for i, c in enumerate(test_clients):
            self.writer.write(f"Client {i + 1}/{len(test_clients)} - {c}")
            swa = False
            if self.server is not None:
                if self.server.swa_model is not None:
                    c.model.load_state_dict(self.server.swa_model.state_dict())
                    swa = True
            loss = c.test(metric, swa=swa)
            if type(metric) == list:
                for m in metric:
                    self.writer.plot_step_loss(m.name, step, loss)
                    self.writer.plot_metric(step, m, str(c), self.ret_score)
                scores.append(metric[0].get_results())
            else:
                self.writer.plot_step_loss(metric.name, step, loss)
                self.writer.plot_metric(step, metric, str(c), self.ret_score)
                scores.append(metric.get_results())
            if type(metric) == list:
                for m in metric:
                    m.reset()
            else:
                metric.reset()
        self.writer.write("Done.")
        return scores

    def test(self, test_clients, metric, step, step_type, max_scores, cl_type='target', prepend=''):

        mean_max_score = sum(max_scores) / len(max_scores)

        if self.server is not None:
            if self.server.swa_model is not None:
                tmp_model = copy.deepcopy(test_clients[0].model.state_dict())

        scores = self.perform_test(metric, test_clients, step)

        metric = metric[0] if isinstance(metric, list) else metric

        ref_scores = [s[self.ret_score] for s in scores]
        mean_score = sum(ref_scores) / len(scores)

        if mean_score > mean_max_score:
            self.writer.write(f"New best result found at {step_type.lower()} {step + 1}")

        for i, score in enumerate(scores):

            ref_client = test_clients[i]
            self.writer.write(f"Test {self.ret_score.lower()} at {step_type.lower()} {step + 1}: "
                              f"{round(score[self.ret_score] * 100, 3)}%")

            if (self.test_plot_counter >= 0 and self.test_plot_counter % 2 == 0) or mean_score > mean_max_score:
                if self.args.save_samples > 0:
                    plot_samples = self.get_plot_samples(ref_client, cl_type=cl_type)
                    for plot_sample in plot_samples:
                        self.writer.plot_samples(metric.name, plot_sample, source=cl_type == 'source', prepend=prepend)
                    self.test_plot_counter = 1
                    self.writer.plot_scores_table(metric, str(ref_client), score, step)
            elif self.test_plot_counter >= 0:
                self.test_plot_counter += 1

        if self.server is not None:
            if self.server.swa_model is not None:
                test_clients[0].model.load_state_dict(tmp_model)

        if mean_score > mean_max_score:
            return ref_scores, True
        return max_scores, False

    @staticmethod
    def get_fake_max_scores(improvement, len_cl):
        if improvement:
            return [0] * len_cl
        return [1] * len_cl
