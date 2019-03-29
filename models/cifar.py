import torch
import torch.nn as nn
from .mnist import Flatten

__all__ = ['cifar10','cifar100']
class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(3,96,3),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Conv2d(96,96,3),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Conv2d(96,96,3, stride=2),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Dropout2d(0.5),

                nn.Conv2d(96,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,3,stride=2),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Dropout2d(0.5),

                nn.Conv2d(192,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,1),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,10,1),

                nn.AvgPool2d(2),
                Flatten()
            )
    def forward(self,x):
        x = self.classifier(x)
        return x

class CIFAR10_DAE(nn.Module):
    def __init__(self):
        super(CIFAR10_DAE, self).__init__()
        self.recon = nn.Sequential(     # b, 3, 32, 32
            nn.ConvTranspose2d(3,2,5,stride=2,padding=0), # b, 2, 67, 67
            nn.ConvTranspose2d(2,1,5,stride=2,padding=0), # b, 2, 137, 137
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Conv2d(1,2,4),       # b, 2, 134, 134 (137-(4-1))
            nn.BatchNorm2d(2),
            nn.MaxPool2d(2,2),       # b, 2, 67, 67 (134/2)
            nn.ReLU(),

            nn.Conv2d(2,3,4),      # b, 3, 64, 64 (67-(4-1))
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2,2),       # b, 3, 32, 32 (64/2)
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.recon(x)
        return x

class CIFAR100():
    def __init__(self):
        print("cifar100")
def cifar10(**kwargs):
    return CIFAR10(),CIFAR10_DAE()

def cifar100():
    return None
