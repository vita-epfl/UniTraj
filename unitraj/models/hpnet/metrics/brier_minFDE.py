import torch
from torchmetrics import Metric

class BrierMinFDE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor,
               prob_best_forecasts: torch.Tensor) -> None:
        errors = torch.norm(predictions - targets, dim=-1)
        minFDE_values = errors[..., -1]
        prob_term = (1.0 - prob_best_forecasts) ** 2
        brier_minFDE_values = minFDE_values + prob_term
        self.sum = self.sum + brier_minFDE_values.sum()
        self.count = self.count + len(brier_minFDE_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count