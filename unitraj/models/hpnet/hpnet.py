"""
HPNet model adapted for UniTraj framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple 
from torch_geometric.data import Batch
from unitraj.models.base_model.base_model import BaseModel
import math

from .modules.backbone import Backbone
from .modules.map_encoder import MapEncoder
from .losses.huber_2d_loss import Huber2DLoss
from .losses.CEloss import CELoss

from .utils.process_data import generate_target
from .utils.process_data import generate_predict_mask

class HPNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        method_config = config.method
        
        # Extract HPNet hyperparameters
        self.hidden_dim = method_config.get('hidden_dim', 128)
        self.num_historical_steps = method_config.get('num_historical_steps', 21)
        self.num_future_steps = method_config.get('num_future_steps', 60)
        self.pos_duration = method_config.get('pos_duration', 20)
        self.pred_duration = method_config.get('pred_duration', 20)
        self.a2a_radius = method_config.get('a2a_radius', 50.0)
        self.l2a_radius = method_config.get('l2a_radius', 50.0)
        self.num_visible_steps = method_config.get('num_visible_steps', 2)
        self.num_modes = method_config.get('num_modes', 6)
        self.num_attn_layers = method_config.get('num_attn_layers', 2)
        self.num_hops = method_config.get('num_hops', 4)
        self.num_heads = method_config.get('num_heads', 8)
        self.dropout = method_config.get('dropout', 0.1)
        self.weight_decay = method_config.get('weight_decay', 1e-4)
        self.warmup_epochs = method_config.get('warmup_epochs', 4)
        self.T_max = method_config.get('T_max', 64)
        self.grad_clip_norm = method_config.get('grad_clip_norm', 5.0)
        self.lr = method_config.get('lr', 3e-4)
        
        # Initialize HPNet submodules
        self.map_encoder = MapEncoder(
            hidden_dim=self.hidden_dim,
            num_hops=self.num_hops,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        self.backbone = Backbone(
            hidden_dim=self.hidden_dim,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            pos_duration=self.pos_duration,
            pred_duration=self.pred_duration,
            a2a_radius=self.a2a_radius,
            l2a_radius=self.l2a_radius,
            num_attn_layers=self.num_attn_layers,
            num_modes=self.num_modes,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Loss functions
        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()
        
    def forward(self, 
                data: Batch,) -> Tuple[Dict, Dict]:
        #print("[HPNet] Batch Data", data)
        lane_embs = self.map_encoder(data=data)
        traj_propose, traj_output, prob_output = self.backbone(data=data, l_embs=lane_embs)
        loss = self._pred_loss(data, traj_propose, traj_output, prob_output)
        output = {}
        
        output['predicted_probability'] = prob_output[:, -1]
        output['predicted_trajectory'] = traj_output[:, -1]
        return output, loss
    
    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        #self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val')
        return loss
    
    
    def _pred_loss(self, data, traj_propose, traj_output, prob_output):
        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]
        errors = (torch.norm(traj_propose[...,:2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(N1,...Nb),H,F,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:,:self.num_historical_steps], self.num_visible_steps)   #[(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]                             #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                        #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                         #[Na,F,2]
        prob = prob_output[predict_mask]                                  #[Na,K]
        targ = target_traj[predict_mask]                                  #[Na,F,2]
        label = best_mode_index[predict_mask]                             #[Na]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask]) 
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])    
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        return loss
    
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    def log_info(self, batch, batch_idx, prediction, status='train'):
        # Extract ground truth from HeteroData batch
        all_positions = batch['agent']['position']  # (total_agents, num_steps, 2)
        all_visible_mask = batch['agent']['visible_mask']  # (total_agents, num_steps)
        agent_indices = batch['agent']['agent_index']  # (B,) - index of predicted agent in each graph
        
        # Get batch information
        batch_ptr = batch['agent'].ptr  # Pointer to split agents by graph
        bs = len(batch_ptr) - 1  # Number of graphs in batch
        
        # Extract ground truth for predicted agents
        gt_traj_list = []
        gt_mask_list = []
        global_agent_indices = []
        for i in range(bs):
            start_idx = batch_ptr[i]
            agent_idx_in_graph = agent_indices[i]
            global_agent_idx = start_idx + agent_idx_in_graph
            global_agent_indices.append(global_agent_idx)
            
            # Extract future trajectory
            future_traj = all_positions[global_agent_idx, self.num_historical_steps:, :]  # (future_len, 2)
            future_mask = all_visible_mask[global_agent_idx, self.num_historical_steps:]  # (future_len,)
            
            gt_traj_list.append(future_traj)
            gt_mask_list.append(future_mask)
        
        global_agent_indices = torch.tensor(global_agent_indices, dtype=torch.long, device=all_positions.device)
        gt_traj = torch.stack(gt_traj_list)  # (B, future_len, 2)
        gt_mask = torch.stack(gt_mask_list)  # (B, future_len)
        
        # Get last valid index for each trajectory
        center_gt_final_valid_idx = torch.sum(gt_mask, dim=-1) - 1  # (B,)
        center_gt_final_valid_idx = torch.clamp(center_gt_final_valid_idx, min=0)
        
        # Extract predictions for predicted agents only
        # predicted_traj: [N, K, F, 2] -> [B, K, F, 2]
        # predicted_prob: [N, K] -> [B, K]
        predicted_traj = prediction['predicted_trajectory'][global_agent_indices]  # (B, K, F, 2)
        predicted_prob = prediction['predicted_probability'][global_agent_indices]  # (B, K)
        predicted_prob = predicted_prob.detach().cpu().numpy()
        
        # Expand for mode comparison
        gt_traj_expanded = gt_traj.unsqueeze(1)  # (B, 1, F, 2)
        gt_mask_expanded = gt_mask.unsqueeze(1)  # (B, 1, F)
        
        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj - gt_traj_expanded, 2, dim=-1)  # (B, K, F)
        ade_losses = torch.sum(ade_diff * gt_mask_expanded, dim=-1) / torch.sum(gt_mask_expanded, dim=-1)  # (B, K)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)  # (B,)
        
        # Calculate FDE losses
        modes = predicted_traj.shape[1]
        center_gt_final_valid_idx_expanded = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)
        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx_expanded).cpu().detach().numpy().squeeze(-1)  # (B, K)
        minfde = np.min(fde, axis=-1)  # (B,)
        
        # Calculate miss rate and brier-FDE
        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob_best = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob_best)
        
        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            'brier_fde': brier_fde
        }
        
        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}
        
        # Log metrics
        for k, v in loss_dict.items():
            self.log(status + "/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])
