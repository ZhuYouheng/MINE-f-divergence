import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MINE(nn.Module):
    def __init__(self, x_size, y_size, proj_size=400):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(x_size, proj_size)
        self.fc2 = nn.Linear(y_size, proj_size)
        self.fc3 = nn.Linear(proj_size, 10)
        self.fc4 = nn.Linear(10, 1)
    
    def forward(self, x, y):
        h = self.fc1(x) + self.fc2(y)
        h = F.relu(h)
        return self.fc4(F.relu(self.fc3(h)))
        #return self.fc3(h)
    
class MINE_1(nn.Module):
    def __init__(self, x_size, proj_size=400):
        super(MINE_1, self).__init__()
        self.fc1 = nn.Linear(x_size, proj_size)
        self.fc3 = nn.Linear(proj_size, 100)
        self.fc4 = nn.Linear(100, 1)
    
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        return self.fc4(F.relu(self.fc3(h)))