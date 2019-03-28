import numpy as np
import torch.nn as nn

__all__ = ['mnist','fashion_mnist']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0],-1)
class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4),
            nn.ReLU(),
            Flatten(),
            nn.Linear(15488, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        #print("mnist")
    def forward(self,x):
        x = self.classifier(x)
        return x

class FMNIST():
    def __init__(self):
        print("fashio mnist")

class MNIST_DAE(nn.Module):
    def __init__(self):
        super(MNIST_DAE,self).__init__()
        self.recon = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 5, stride=2, padding=0),  # b, 1, 67, 67
            nn.ConvTranspose2d(1, 1, 5, stride=2, padding=0),  # b, 1, 137, 137
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Conv2d(1, 1, 4),  # b, 2, 134, 134 (137-(4-1))
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2, 2),  # b, 2, 67, 67 (134/2)
            nn.ReLU(),

            nn.Conv2d(1, 1, 4),  # b, 3, 64, 64 (67-(4-1))
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2, 2),  # b, 3, 32, 32 (64/2)
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.recon(x)
        return x




def mnist(**kwargs):
    return MNIST(), MNIST_DAE()

def fashion_mnist():
    return None

