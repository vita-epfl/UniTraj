import torch
from torchmetrics import Metric

class MR(Metric):
    def __init__(self, 
                 threshold: float=2.0,
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> None:
        errors = torch.norm(predictions[:,-1] - targets[:,-1], dim=-1)
        MR_values = errors > self.threshold
        self.sum = self.sum + MR_values.sum()
        self.count = self.count + len(MR_values)
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count