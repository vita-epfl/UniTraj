from typing import Optional

import torch
from torchmetrics import Metric

from unitraj.models.smart.metrics.utils import topk
from unitraj.models.smart.metrics.utils import valid_filter


class minMultiFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minMultiFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 eval_timestep: int = 60,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.eval_timestep = eval_timestep

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        eval_timestep = min(self.eval_timestep, pred.shape[1]) - 1
        # last_idx = eval_timestep - 1

        # pred_last = pred[:, last_idx, :]
        # target_last = target[:, last_idx, :]

        # non_zero_mask = ~((pred_last[:, 0] == 0) & (pred_last[:, 1] == 0))
        # valid = valid_mask[:, last_idx]
        # combined_mask = valid & non_zero_mask

        # error = torch.norm(pred_last - target_last, p=2, dim=-1)
        # weighted_error = error * combined_mask

        # if weighted_error.sum() < 200:
        #     self.sum += weighted_error.sum()
        #     self.count += combined_mask.sum()
        self.sum += ((torch.norm(pred[:, eval_timestep-1:eval_timestep] - target[:, eval_timestep-1:eval_timestep], p=2, dim=-1) *
                      valid_mask[:, eval_timestep-1].unsqueeze(1)).sum(dim=-1)).sum()
        self.count += valid_mask[:, eval_timestep-1].sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
