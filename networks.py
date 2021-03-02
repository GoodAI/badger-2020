import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *

class ControlNet(nn.Module):
    def __init__(self, CODE, HIDDEN, LAYERS, SENSORS, ACTIONS, continuous = False):
        super().__init__()
        
        self.ll = nn.ModuleList([nn.Linear(SENSORS + CODE, HIDDEN)] + \
            [nn.Linear(HIDDEN, HIDDEN) for i in range(LAYERS-2)])
        
        for l in self.ll:
            nn.init.orthogonal_(l.weight, gain=sqrt(2))
        self.out = nn.Linear(HIDDEN, ACTIONS)
        self.continuous = continuous
        
    def forward(self, x, a):
        z = torch.cat([x,a], 1)
        
        for l in self.ll:
            z = F.relu(l(z))
        
        if self.continuous:
            z = F.tanh(self.out(z))
        else:
            z = F.softmax(self.out(z), dim=1)
        
        return z
    
    def get_action(self, state, code):
        if self.continuous:
            p = self.forward(state, code).cpu().detach().numpy()[0]
            return p
        else:
            p = self.forward(state, code).cpu().detach().numpy()[0]
            return np.random.choice(np.arange(p.shape[0]), p=p)
