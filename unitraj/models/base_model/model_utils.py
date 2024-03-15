from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torch import nn
from tqdm import tqdm
from typing_extensions import Protocol


class EmbeddingInitializer(Protocol):
    """A callable that takes an embedding size and initializes an embedding layer
    """

    def __call__(self, embedding_size: int) -> nn.Module:
        """ Initialize the embedding layer.

        Args:
            embedding_size (int): Size of the embedding layer output.

        Returns:
            nn.Module: The initialized embedding layer.
        """
        ...


@dataclass
class EmbeddingPair:
    """
    Datasets    | Embedding initializer
    A           |   X
    B           |   Y
    C           |   X
    D           |   Z
    E           |   X
    ---------------------------

    [A, C, E] -> X
    [B] -> Y
    [D] -> Z
    """
    datasets: List[str]
    embedding_initializer: EmbeddingInitializer


import matplotlib.pyplot as plt


def draw_scene(ego, agents, map):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Function to interpolate between green and blue colors
    def interpolate_color(t, total_t):
        return (0, 1 - t / total_t, t / total_t)  # Interpolating between green and blue

    def interpolate_color_ego(t, total_t):
        return (1 - t / total_t, 0, t / total_t)

    # Function to draw lines with a validity check
    def draw_line_with_mask(point1, point2, color, line_width=4):
        if point1[2] and point2[2]:  # Check if both points are valid
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    # Plot the map with mask check
    for lane in map:
        for i in range(len(lane) - 1):
            draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
            else:
                color = interpolate_color(t, total_t)
            draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw ego vehicle trajectory
    draw_trajectory(ego, line_width=5, ego=True)

    # Draw trajectories for other agents
    for i in range(agents.shape[1]):
        draw_trajectory(agents[:, i, :], line_width=5)

    # Set labels, limits, and other properties
    vis_range = 50
    # ax.legend()
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    ax.set_aspect('equal')
    ax.axis('off')

    # Return the axes object
    return ax


class FisherInformationCallback(Callback):
    def __init__(self):
        return

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        fisher_information = {}
        for name, param in pl_module.named_parameters():
            fisher_information[name] = torch.zeros_like(param, device=param.device)

        pl_module.eval()
        # with torch.no_grad():
        for idx, batch in enumerate(trainer.train_dataloader):
            pl_module.zero_grad()
            model_input, ground_truth = batch['data'], batch['gt']
            for key in model_input.keys():
                try:
                    model_input[key] = model_input[key].float().to(pl_module.device)
                except:
                    continue
            ground_truth = ground_truth.float().to(pl_module.device)
            output = pl_module._forward(model_input)

            loss, loss_dict = pl_module.criterion(output, ground_truth)
            loss.backward()
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad ** 2 / len(trainer.train_dataloader)

        # Save or process the Fisher Information as needed
        # For example, you can attach it to the pl_module
        checkpoint['fisher_information'] = fisher_information
        # save current model params as optimal params
        optimal_params = OrderedDict({})
        for name, param in pl_module.named_parameters():
            optimal_params[name] = param.clone().detach()
        checkpoint['optimal_params'] = optimal_params


class SmartSamplerCallback(Callback):

    def __init__(self, cfg):
        self.config = cfg
        return

    def on_train_start(self, trainer, pl_module):
        pl_module.eval()
        sampler = trainer.train_dataloader.sampler

        sampler.reset()
        sample_mode = self.config['sample_mode']
        sample_num = self.config['sample_num']

        if sample_mode == 'entropy':
            selected_indx = self.entropy_sampler(trainer, pl_module, sample_num)
        elif sample_mode == 'random':
            selected_indx = self.random_sampler(trainer, pl_module, sample_num)
        elif sample_mode == 'fisher':
            selected_indx = self.fisher_sampler(trainer, pl_module, sample_num)
        else:
            raise NotImplementedError

        sampler.set_idx(selected_indx)

    def random_sampler(self, trainer, pl_module, sample_num):
        all_idx = trainer.train_dataloader.sampler.idx
        sample_num = min(sample_num, len(all_idx))
        random_idx = np.random.choice(all_idx, sample_num, replace=False).tolist()
        return random_idx

    def fisher_sampler(self, trainer, pl_module, sample_num):
        pl_module.eval()
        idx_list = []
        metrics_list = []

        fisher_information = pl_module.fisher_information

        for idx, batch in tqdm(enumerate(trainer.train_dataloader)):
            # batch_list = create_pseudo_batches(batch_all)

            pl_module.zero_grad()
            model_input, ground_truth = batch['data'], batch['gt']
            for key in model_input.keys():
                try:
                    model_input[key] = model_input[key].float().to(pl_module.device)
                except:
                    continue
            ground_truth = ground_truth.float().to(pl_module.device)
            output = pl_module._forward(model_input)
            loss, loss_dict = pl_module.criterion(output, ground_truth)
            loss.backward()

            gaps = []
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_weight = param.grad.view(-1) ** 2
                    original_weight = fisher_information[name].view(-1)
                    grad_weight_softmax = F.softmax(grad_weight)
                    original_weight_softmax = F.softmax(original_weight)
                    epsilon = 1e-10  # Small constant for numerical stability
                    grad_weight_softmax += epsilon
                    gap = F.kl_div(grad_weight_softmax.log(), original_weight_softmax.to(pl_module.device),
                                   reduction='batchmean')
                    gaps.append(gap)

            # gaps = []
            # for name, param in pl_module.named_parameters():
            #     if param.grad is not None:
            #         grad_vector = param.grad.view(1,-1)
            #         fisher_vector = fisher_information[name].view(1,-1).to(pl_module.device)
            #         # Normalizing vectors
            #         grad_norm = F.normalize(grad_vector)
            #         fisher_norm = F.normalize(fisher_vector)
            #
            #         # Cosine similarity and Euclidean distance
            #         #cos_sim = cosine_similarity(grad_norm, fisher_norm)
            #         euc_dist = euclidean_distance(grad_norm, fisher_norm)
            #
            #         # Combining metrics (example: simple average)
            #         #gap = (cos_sim + euc_dist) / 2
            #         gaps.append(euc_dist)

            gap_for_this_batch = torch.stack(gaps).mean()
            # repeat gap_for_this_batch n times, n is the batch size
            gap_for_this_batch = gap_for_this_batch.repeat(batch['gt'].shape[0])
            metrics_list.append(gap_for_this_batch)
            idx_list.append(batch['idx'])

        metrics = torch.cat(metrics_list)
        idx = torch.cat(idx_list)
        _, sorted_idx = torch.sort(metrics, descending=True)
        sorted_idx = idx[sorted_idx]

        return sorted_idx[:sample_num].cpu().tolist()

    def entropy_sampler(self, trainer, pl_module, sample_num):
        pl_module.eval()
        # sampler = trainer.train_dataloader.sampler
        idx_list = []
        metrics_list = []

        with torch.no_grad():
            for idx, batch in enumerate(trainer.train_dataloader):
                pl_module.zero_grad()
                model_input, ground_truth = batch['data'], batch['gt']
                for key in model_input.keys():
                    try:
                        model_input[key] = model_input[key].float().to(pl_module.device)
                    except:
                        continue
                output = pl_module._forward(model_input)
                modes_pred = output['modes_pred']
                epsilon = 1e-10
                modes_pred = modes_pred + epsilon
                # Calculate the entropy
                log_preds = torch.log(modes_pred)
                entropy = -torch.sum(modes_pred * log_preds, dim=1)

                idx_list.append(batch['idx'])
                metrics_list.append(entropy)

        metrics = torch.cat(metrics_list)
        idx = torch.cat(idx_list)

        metrics = metrics * 5
        metrics = F.softmax(metrics, dim=0)
        # metrics is the sample weight
        sampled_idx = torch.multinomial(metrics, sample_num, replacement=False)

        # _, sorted_idx = torch.sort(metrics, descending=True)
        # sorted_idx = idx[sorted_idx]
        # get sample_num evenly spaced indices
        # sample_idx = np.linspace(0, len(sorted_idx)-1, sample_num).astype(int)
        return idx[sampled_idx].cpu().tolist()


def create_pseudo_batches(original_batch):
    pseudo_batches = []

    # Assuming the batch size of the original batch is greater than 0
    batch_size = original_batch['gt'].shape[0]

    for i in range(batch_size):
        # Creating a new batch for each item
        pseudo_batch = {}
        for key, value in original_batch.items():
            if isinstance(value, torch.Tensor):
                # For tensors, just take the i-th element
                pseudo_batch[key] = value[i:i + 1]  # Keeping the tensor dimensions
            elif isinstance(value, list):
                # For lists, take the i-th element (assumes a list of tensors)
                pseudo_batch[key] = [value[i]]
            elif isinstance(value, dict):
                # For dictionaries, recurse
                pseudo_batch[key] = {k: v[i:i + 1] if isinstance(v, torch.Tensor) else v[i] for k, v in value.items()}
            else:
                raise TypeError(f"Unsupported type for batching: {type(value)}")

        pseudo_batches.append(pseudo_batch)

    return pseudo_batches


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2)


def euclidean_distance(vec1, vec2):
    return torch.norm(vec1 - vec2, p=2)
