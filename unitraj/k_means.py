import pickle

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from datasets import build_dataset
from utils.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def cluster(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    train_set = build_dataset(cfg)

    train_loader = DataLoader(
        train_set, batch_size=1024, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    vehicle_last_pos_list = []
    pedestrian_last_pos_list = []
    cyclist_last_pos_list = []
    for batch in train_loader:
        inputs = batch['input_dict']
        ego_gt = inputs['center_gt_trajs'][..., :2]
        final_gt = torch.gather(ego_gt, 1,
                                inputs['center_gt_final_valid_idx'].type(torch.int64).unsqueeze(-1).unsqueeze(
                                    -1).repeat(1, 1, 2)).squeeze(1)
        vehicle_last_pos_list.append(final_gt[inputs['center_objects_type'] == 1])
        pedestrian_last_pos_list.append(final_gt[inputs['center_objects_type'] == 2])
        cyclist_last_pos_list.append(final_gt[inputs['center_objects_type'] == 3])

    vehicle_last_pos_array = np.concatenate(vehicle_last_pos_list)
    pedestrian_last_pos_array = np.concatenate(pedestrian_last_pos_list)
    cyclist_last_pos_array = np.concatenate(cyclist_last_pos_list)

    kmeans = KMeans(n_clusters=64)

    dict_clusters = {}
    if len(vehicle_last_pos_array) > 0:
        vehicle_cluster = kmeans.fit(vehicle_last_pos_array)
        dict_clusters['VEHICLE'] = vehicle_cluster.cluster_centers_

    if len(pedestrian_last_pos_array) > 0:
        pedestrian_cluster = kmeans.fit(pedestrian_last_pos_array)
        dict_clusters['PEDESTRIAN'] = pedestrian_cluster.cluster_centers_

    if len(cyclist_last_pos_array) > 0:
        cyclist_cluster = kmeans.fit(cyclist_last_pos_array)
        dict_clusters['CYCLIST'] = cyclist_cluster.cluster_centers_

    with open(cfg['MOTION_DECODER']['INTENTION_POINTS_FILE'], 'wb') as handle:
        pickle.dump(dict_clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cluster()
