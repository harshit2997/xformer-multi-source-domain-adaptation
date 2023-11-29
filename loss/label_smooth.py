import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, temperature=0.07):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.temperature = temperature

    def forward(self, x, target_logits):
        target_logits = F.softmax(target_logits / self.temperature, dim=1)
        log_likelihood = -F.log_softmax(x, dim=1)
        loss = torch.mean(torch.sum(log_likelihood*target_logits, dim=1))
        return loss
