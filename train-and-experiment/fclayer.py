import torch
import torch.nn as nn
from typing import Callable
from torch.nn.functional import linear, normalize

class FCSoftmax(nn.Module):
    def __init__(self, margin_softmax: Callable, embed_size: int, num_classes: int):
        super(FCSoftmax, self).__init__()
        self.margin_softmax = margin_softmax
        self.weights = torch.nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        nn.init.xavier_uniform_(self.weights)
    
    def forward(self, embed_vec: torch.Tensor, labels: torch.Tensor):
        # logits is "cos(theta)"
        logits = linear(normalize(embed_vec), normalize(self.weights)).clamp(-1,1)
        # arcface: cos(target_theta + margin), cosface: cos(target_theta)-margin
        logits = self.margin_softmax(logits, labels)
        return logits

    def state_dict(self):
        return self.weights.data

    def load_state_dict(self, checkpoint):
        self.weights.data = checkpoint.to(self.weights.data.device)