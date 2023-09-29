import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MINE(nn.Module):
    def __init__(self, x_size, y_size, proj_size=16):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(x_size, proj_size)
        self.fc2 = nn.Linear(y_size, proj_size)
        self.fc3 = nn.Linear(proj_size, 1)
    
    def forward(self, x, y):
        h = self.fc1(x) + self.fc2(y)
        h = F.relu(h)
        return self.fc3(h)
    
