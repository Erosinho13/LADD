from copy import deepcopy
from federated.trainers import FdaTrainer, OracleTrainer
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statistics import mode
import re
import pickle
from metrics import StreamSegMetrics


class LADDTrainer(FdaTrainer):

    def __init__(self, args, writer, device, rank, world_size):

        super(OracleTrainer, self).__init__(args, writer, device, rank, world_size)

        writer.write(f'LADD Cluster - Initializing style transfer module in clients...')
        self.cluster_path = None
        self.k_means_model = None
        self.swa_teacher_model = None
        if self.args.swa_start != -1:
            self.swa_n = 0

        if self.args.swa_teacher_start != -1:
            self.swa_teacher_n = 0

        if self.args.style_clusters_dir is not None:
            if not os.path.exists(self.args.style_clusters_dir):
                os.mkdir(self.args.style_clusters_dir)

            if self.args.force_k == 0:
                cluster_relative_path = f"_cluster_{self.args.clients_type}.json"
            else: 
                cluster_relative_path =  f"_cluster_{self.args.clients_type}_{self.args.force_k}.json"
            self.cluster_path = os.path.join(self.args.style_clusters_dir,
                                            f"{self.args.target_dataset}_b{self.args.fda_b}" + cluster_relative_path)

        for clients in (self.source_train_clients, self.source_test_clients):
            for c in clients:
                c.is_source_client()

        self.styles_mapping = {}
        self.cluster_mapping = {}
        if self.cluster_path and os.path.exists(self.cluster_path):
            writer.write(f'Load fda clustering from {self.cluster_path}')
            with open(self.cluster_path) as json_file:
                self.cluster_mapping = json.load(json_file)

            self.k_size = len(self.cluster_mapping.keys())
            if self.args.force_k == 0:
                k_means_relative_path = "_model.pkl"
            else:
                k_means_relative_path = f"_model_{self.args.force_k}.pkl"
            self.k_means_model = pickle.load(open(self.cluster_path.split(".json")[0] + k_means_relative_path, "rb"))

        else:
            writer.write(f'Fda clustering {self.cluster_path} not present..')
            writer.write('Send style from clients to server')
            self.set_server_style()
            writer.write('Cluster clients based on style')
            self.cluster_styles()

        for client in self.target_train_clients:
            for cluster_id in self.cluster_mapping.keys():
                if client.id in self.cluster_mapping[cluster_id]:
                    client.cluster_id = int(cluster_id)
                    break

        self.server.num_clusters = len(self.cluster_mapping.keys())

        self.writer.write(f"Clusters found: {self.cluster_mapping}")

        correct = 0
        for cluster_id in self.cluster_mapping.keys():
            cities = [re.sub("\d+", "", client_name) for client_name in self.cluster_mapping[cluster_id]]
            city_ground_truth = mode(cities)
            for client_name in self.cluster_mapping[cluster_id]:
                if city_ground_truth in client_name:
                    correct += 1

        accuracy = correct / len(self.target_train_clients)
        self.writer.write(f'clustering accuracy {accuracy}')
        writer.write(f'Done')

        self.target_test_clients[0].find_test_images_cluster(k_means_model=self.k_means_model)
        self.writer.write(f"Test image cluster mapping: {self.target_test_clients[0].test_images_clusters}")
        if args.test_only_with_global_model:
            self.perform_test = super().perform_test

        self.server.batch_norm_dict = {cluster_id: None for cluster_id in range(self.server.num_clusters)}

    def set_metrics(self, writer, num_classes):
        writer.write('Setting up metrics...')
        target_test = StreamSegMetrics(num_classes, 'target_test') if self.args.test_only_with_global_model else \
            [StreamSegMetrics(num_classes, 'target_test_global'), StreamSegMetrics(num_classes, 'target_test_cluster')]

        metrics = {
            'server_train': StreamSegMetrics(num_classes, 'server_train'),
            'uda_train': StreamSegMetrics(num_classes, 'uda_train'),
            'server_test': StreamSegMetrics(num_classes, 'server_test'),
            'target_test': target_test,
            'target_eval': StreamSegMetrics(num_classes, 'target_eval')
        }
        writer.write('Done.')
        return metrics

    def set_server_style(self):

        for client in self.target_train_clients:
            self.server.styleaug.add_style(client.dataset, multiple_styles=self.args.multiple_styles, name=client.id)

        self.source_train_clients[0].loader.dataset.set_style_tf_fn(self.server.styleaug.apply_style)

        styles = [style.tolist() for style in self.server.styleaug.styles]
        self.styles_mapping = {"style": styles, "id": self.server.styleaug.styles_names}

    def cluster_styles(self):
        styles_flat = np.array(self.styles_mapping["style"]).reshape(len(self.styles_mapping["style"]), -1)
        model_list = []
        res_list = []
        score_list = []
        if self.args.force_k == 0:
            k_list = list(range(4, 20))
        else:
            k_list = [self.args.force_k]
        for k_size in k_list:
            model = KMeans(n_clusters=k_size, n_init=10).fit(styles_flat)
            model_list.append(model)
            res_list.append(model.labels_)
            score_list.append(silhouette_score(styles_flat, model.labels_))

        best_id = np.argmax(score_list)
        self.k_means_model = model_list[best_id]
        if self.args.force_k == 0:
            k_means_relative_path = "_model.pkl"
        else:
            k_means_relative_path = f"_model_{self.args.force_k}"
        pickle.dump(self.k_means_model, open(self.cluster_path.split(".json")[0] + k_means_relative_path, "wb"))
        self.k_size = k_list[best_id]
        self.writer.write(f"best k {self.k_size}")
        self.writer.write(f"best silhouette_score {score_list[best_id]}")
        for cluster_id in range(self.k_size):
            self.cluster_mapping[cluster_id] = [self.styles_mapping["id"][i][0]
                                                for i, val in enumerate(res_list[best_id])
                                                if val == cluster_id]
        if self.cluster_path is not None:
            with open(self.cluster_path+"utf8-aware", "w", encoding='utf-8') as fp:
                json.dump(self.cluster_mapping, fp, ensure_ascii=False)
            self.writer.write(f"Saved fda style clusters to {self.cluster_path}")

    def setup_swa_teacher_model(self, swa_ckpt=None):
        self.swa_teacher_model = [deepcopy(self.model) for _ in range(self.k_size)]

    def set_client_teacher(self, r, model):

        if r % self.args.teacher_step == 0 and not self.args.teacher_upd_step:

            self.writer.write(f"round {r}, setting new teacher...")

            if self.args.teacher_kd_step == -1 and self.args.lambda_kd > 0:
                self.writer.write(f"Setting kd teacher too...")

            if self.args.swa_teacher_start != -1 and r + 1 > self.args.swa_teacher_start and \
                    ((r - self.args.swa_teacher_start) // self.args.teacher_step) % self.args.swa_teacher_c == 0:
                swa_teacher_models = []
                self.writer.write(f"Number of models: {self.swa_teacher_n}")
                for swa_teacher_model in self.swa_teacher_model:
                    swa_teacher_models.append(self.update_swa_teacher_model(1.0 / (self.swa_teacher_n + 1), model,
                                              swa_teacher_model=swa_teacher_model))
                self.swa_teacher_n += 1
                self.swa_teacher_model = swa_teacher_models

            for c in self.target_train_clients:
                if hasattr(c.criterion, 'set_teacher'):
                    if self.args.fw_task == "ladd" and self.server.clusters_models:
                        if self.swa_teacher_model is not None:
                            c.criterion.set_teacher(self.swa_teacher_model[c.cluster_id])
                        else:
                            model.load_state_dict(self.server.clusters_models[c.cluster_id])
                            c.criterion.set_teacher(model)
                else:
                    break

            if self.args.count_classes_teacher_step != -1:
                if (r // self.args.teacher_step) % self.args.count_classes_teacher_step == 0:
                    self.writer.write("Updating sampling probs...")
                    self.server.count_classes()
                    self.writer.write("Done.")

            self.writer.write(f"Done.")
