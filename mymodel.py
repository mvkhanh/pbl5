import torch.nn as nn
import torch
from huggingface_hub import hf_hub_download

from utils.uniformer import uniformer_base
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.uniformer_ = uniformer_base()
        model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
        state_dict = torch.load(model_path, map_location='cpu')
        self.uniformer_.load_state_dict(state_dict)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )
        for param in self.uniformer_.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.uniformer_.forward_features(x)
        return self.mlp(x)