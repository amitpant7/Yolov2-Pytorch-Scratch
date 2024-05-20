from torch import nn
import torch

def initialize(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        with torch.no_grad():
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

def intializer(model):
    return model.apply(initialize)