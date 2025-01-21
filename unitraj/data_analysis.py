import torch

torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from datasets import build_dataset
from datasets.common_utils import trajectory_correspondance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def data_analysis(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    train_set = build_dataset(cfg)
    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    wandb.init(project="unitraj", name=cfg.exp_name)
    type_results = {}
    kalman_results = {}
    vehicle_sum = 0
    pedestrian_sum = 0
    cyclist_sum = 0
    for batch in train_loader:
        obj_types = batch['input_dict']['center_objects_type']
        vehicle_sum += np.sum(obj_types == 1)
        pedestrian_sum += np.sum(obj_types == 2)
        cyclist_sum += np.sum(obj_types == 3)

        input = batch['input_dict']
        dataset_names = input['dataset_name']
        trajectory_types = input['trajectory_type']
        kalman_diffs = input['kalman_difficulty']
        unique_dataset_names = np.unique(dataset_names)

        for dataset_name in unique_dataset_names:
            if dataset_name not in type_results:
                type_results[dataset_name] = {}
            if dataset_name not in kalman_results:
                kalman_results[dataset_name] = []

            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            trajectory_types_for_this_dataset = trajectory_types[batch_idx_for_this_dataset]
            for traj_type in range(8):
                batch_idx_for_traj_type = np.where(trajectory_types_for_this_dataset == traj_type)[0]
                traj_type = trajectory_correspondance[traj_type]
                if len(batch_idx_for_traj_type) > 0:
                    if traj_type not in type_results[dataset_name]:
                        type_results[dataset_name][traj_type] = 0
                    type_results[dataset_name][traj_type] += len(batch_idx_for_traj_type)

            kalman_diffs_for_this_dataset = kalman_diffs[batch_idx_for_this_dataset]
            kalman_results[dataset_name].append(kalman_diffs_for_this_dataset)
    count_dict = {'vehicle': vehicle_sum, 'pedestrian': pedestrian_sum, 'cyclist': cyclist_sum}
    count_df = pd.DataFrame([count_dict], columns=list(count_dict.keys()))
    wandb.log({"Count of each type": wandb.Table(dataframe=count_df)})

    for dataset_name in kalman_results:
        kalman_results[dataset_name] = np.concatenate(kalman_results[dataset_name], axis=0)

    # Determine the global range of bins needed
    all_values = np.concatenate([data[:, -1] for data in kalman_results.values()])
    all_values = all_values[(all_values < 100) & (all_values > 0)]
    bin_width = 10
    global_bins = np.arange(start=0, stop=all_values.max() + bin_width, step=bin_width)

    # Prepare data for grouped bar chart
    bin_centers = 0.5 * (global_bins[:-1] + global_bins[1:])
    bar_data = pd.DataFrame(index=bin_centers)

    for dataset_name, kalman_diffs in kalman_results.items():
        kalman_diffs = kalman_diffs[:, -1]
        kalman_diffs = kalman_diffs[(kalman_diffs < 100) & (kalman_diffs > 0)]
        counts, _ = np.histogram(kalman_diffs, bins=global_bins)
        bar_data[dataset_name] = counts / counts.sum() * 100  # Convert to percentage

    # Melt the DataFrame for consistency with the other chart
    melted_data = bar_data.reset_index().melt(id_vars='index', var_name='Dataset', value_name='Count')
    melted_data.rename(columns={'index': 'Value Range'}, inplace=True)

    # Create a grouped bar chart
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Value Range', y='Count', hue='Dataset', data=melted_data, palette="viridis", edgecolor='black')

    # Customize the plot aesthetics
    plt.xlabel('Value Range', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Kalman Difficulty Comparison (Percentage)', fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    wandb.log({f"Kalman Difficulty for {dataset_name}": wandb.Image(plt)})

    # Clear the figure
    plt.clf()

    # Prepare a DataFrame to store the percentages
    all_data_percentage = pd.DataFrame()

    for dataset_name, type_counts in type_results.items():
        # Convert type_counts to DataFrame
        df = pd.DataFrame(list(type_counts.items()), columns=['Trajectory Type', 'Count'])

        # Calculate the total count for the dataset
        total_count = df['Count'].sum()

        # Calculate the percentage for each trajectory type
        df[f'{dataset_name}'] = (df['Count'] / total_count) * 100

        # Merge with the all_data_percentage DataFrame
        if all_data_percentage.empty:
            all_data_percentage = df[['Trajectory Type', f'{dataset_name}']]
        else:
            all_data_percentage = all_data_percentage.merge(df[['Trajectory Type', f'{dataset_name}']],
                                                            on='Trajectory Type', how='outer')

    # Replace NaN values with 0
    all_data_percentage.fillna(0, inplace=True)

    # Melt the DataFrame to long format for seaborn
    melted_data_percentage = all_data_percentage.melt(id_vars='Trajectory Type', var_name='Dataset',
                                                      value_name='Percentage')

    # Create a grouped bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Trajectory Type', y='Percentage', hue='Dataset', data=melted_data_percentage, palette="viridis",
                edgecolor='black')

    # Customize the plot aesthetics
    plt.xlabel('Trajectory Type', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Trajectory Types Comparison (Percentage)', fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')

    wandb.log({f"Trajectory Types for {dataset_name}": wandb.Image(plt)})

    # Clear the figure after logging
    plt.clf()


if __name__ == '__main__':
    data_analysis()
