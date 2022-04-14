from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class BasicLoss(nn.Module):

    def __init__(self, mse_weight: float = 1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, model_input, model_output) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, gt_depth = model_input
        pred = model_output

        if self.mse_weight > 0:
            mse_loss = self.mse_loss(pred, gt_depth)

        loss = (self.mse_weight) * mse_loss
        loss_dict = {"loss": loss.item(), "mse" : mse_loss.item()}
        return loss, loss_dict
