import torch.nn as nn
import torch
from huggingface_hub import hf_hub_download

from utils.uniformer import uniformer_base
class MyModel(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        return self.mlp(x)