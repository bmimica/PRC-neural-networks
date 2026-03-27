import torch
import torch.nn as nn

# chooses automatically if do gpu or cpu calculations
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# to do a checkpoint
from torch.utils.checkpoint import checkpoint as cp
save_memory = False

    


class Sequential(nn.Module):
    def __init__(self, layers):
        
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x