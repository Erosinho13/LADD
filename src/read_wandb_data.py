import os
import wandb
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from celluloid import Camera

DATA_DIR = "wandb_data"


class WandbData:
    def __init__(self, args):
        self.args = args
        self.wandb_run_path = os.path.join(args.wandb_id, args.project_name, args.wandb_id)
        self.dl_run_path = os.path.join(DATA_DIR, args.project_name, args.wandb_id)
        self.target_dataset = args.project_name.split("gta5")[-1].split("_")[-1]
        self.table_path = None
        self.step_iou = None

    def download(self):
        print(f"Saving data in {self.dl_run_path}")
        run = wandb.Api().run(self.wandb_run_path)
        os.makedirs(self.dl_run_path, exist_ok=True)
        for file in run.files():
            print(file)
            file.download(root=self.dl_run_path)

    def read_tables(self):
        self.table_path = os.path.join(self.dl_run_path, "media", "table")
        self.step_iou = {}
        for file in os.listdir(self.table_path):
            table_name = file[:-31].split("scores")[0][:-1]
            if table_name not in self.step_iou.keys():
                self.step_iou[table_name] = {}
            step = int(file[:-31].split("_")[-2])
            with open(os.path.join(self.table_path, file), 'r') as f:
                self.step_iou[table_name][step] = json.load(f)

    def plot_iou_table(self):
        if self.target_dataset in ["cityscapes", "mapillary", "crosscity"]:
            labels = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                      "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                      "motorbike", "bicycle"]
            if self.target_dataset == "crosscity":
                labels = ["road", "sidewalk", "building", "traffic light", "traffic sign", "vegetation", "sky",
                          "person", "rider", "car", "bus", "motorcycle", "bicycle"]

            for table_name in self.step_iou.keys():
                fig, ax = plt.subplots(1, 1)
                camera = Camera(fig)
                for key in sorted(self.step_iou[table_name].keys()):
                    value = self.step_iou[table_name][key]
                    iou_index = value["columns"].index("IoU")
                    iou = [float(v[iou_index]) for v in value["data"]]
                    miou = np.mean(iou)
                    iou_dict = {k: float(v) for k, v in zip(labels, iou)}
                    iou_dict = {**iou_dict, "mIoU": miou}
                    iou.append(float(miou))
                    sns.barplot(x=list(iou_dict.values()), y=list(iou_dict.keys()), color='navy')
                    plt.legend(labels=[f"mIoU = {miou: .4f}, step {key}"])
                    plt.xlim((None, 1))
                    plt.tight_layout()
                    camera.snap()

                anim = camera.animate(interval=250)
                anim.save(os.path.join(self.dl_run_path, f"{table_name}_iou.mp4"), dpi=300)


if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="example federated_gta5_crosscity")
    parser.add_argument("--wandb_id", type=str, help="wandb id of the run")
    parser.add_argument("--download", action="store_true", default=False, help="download data if set")
    args = parser.parse_args()
    wandb_data = WandbData(args)
    if args.download:
        wandb_data.download()
    wandb_data.read_tables()
    wandb_data.plot_iou_table()
