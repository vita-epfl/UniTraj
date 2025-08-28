import torch

torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
from datasets.common_utils import trajectory_correspondance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
import hydra
from omegaconf import OmegaConf
from unitraj.vis import visualize_scenario
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

SHOW_ROADLINES = False
SHOW_PED_XINGS = True


@hydra.main(version_base=None, config_path="configs", config_name="config")
def visualize(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    cfg['eval'] = True

    model_class = build_model(cfg).__class__
    model = model_class.load_from_checkpoint(cfg.ckpt_path, config=cfg)
    model = model.cuda()
    train_set = build_dataset(cfg)
    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices), 1)
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    wandb.init(project="unitraj", name=cfg.exp_name)
    predict = True
    

    for batch in train_loader:
        input = batch['input_dict']
        input_cpu = input.copy()
        for k in input.keys():
            if torch.is_tensor(input[k]):
                input[k] = input[k].cuda()
        if predict:
            with torch.no_grad():
                batch_pred = model.predict(batch)
        for idx_in_batch in range(batch['batch_size']):
            if predict:
                visualize_scenario(input_cpu, idx_in_batch, timestep=cfg.get('past_len'), show_history=True, show_future=True, show_map=True, 
                                   prediction=batch_pred['predicted_trajectory'].cpu(), predicition_probs=batch_pred['predicted_probability'].cpu(), 
                                   show_roadlines=SHOW_ROADLINES, show_ped_xings=SHOW_PED_XINGS)
            else:
                visualize_scenario(input_cpu, idx_in_batch, timestep=cfg.get('past_len'), show_history=True, show_future=True, show_map=True, 
                                   show_roadlines=SHOW_ROADLINES, show_ped_xings=SHOW_PED_XINGS)
            plt.show()


if __name__ == '__main__':
    visualize()
