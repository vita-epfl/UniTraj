import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = F.nll_loss(torch.log(predictions), targets)
        return loss