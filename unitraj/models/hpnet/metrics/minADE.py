import torch
from torchmetrics import Metric

class MinADE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        errors = torch.norm(predictions - targets, dim=-1)
        minADE_values = errors.mean(dim=-1)
        self.sum = self.sum + minADE_values.sum()
        self.count = self.count + len(minADE_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count