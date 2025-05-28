from typing import Optional

import torch
from torchmetrics import Metric

from unitraj.models.smart.metrics.utils import topk
from unitraj.models.smart.metrics.utils import valid_filter


class TokenCls(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(TokenCls, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None) -> None:
        target = target[..., None]
        acc = (pred[:, :self.max_guesses] == target).any(dim=1) * valid_mask
        self.sum += acc.sum()
        self.count += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
