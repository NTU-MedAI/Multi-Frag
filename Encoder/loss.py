import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
class CrossEntropy(nn.Module):
    def forward(self, input, target):
        # print(input)
        # print(target)
        scores = torch.sigmoid(input)
        target_active = (target == 1).float()  # from -1/1 to 0/1
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        b=loss_terms.sum()/len(loss_terms)
        return b
