import torch
import torch.nn as nn

import os

# chooses automatically if do gpu or cpu calculations
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# to do a checkpoint
from torch.utils.checkpoint import checkpoint as cp
save_memory = False


class Sequential(nn.Module):
    def __init__(self, layers, label):
        
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.label = label
        self.path = None

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self):
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
          
        model_file = os.path.join(models_dir, f"model_{self.label}.pth")
        torch.save(self, model_file)
        self.path = model_file
      

class LeakyResidualConnector(nn.Module):
    def __init__(self, layers, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(size) # do i need this ? 
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, *outputs):
        summed_outputs = 0
        for out in outputs:
            summed_outputs += self.norm(self.dropout(out.view_as(x)))
            
        return x + summed_outputs