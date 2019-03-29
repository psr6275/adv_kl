import torch
import torch.nn as nn
from .mnist import Flatten
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

__all__ = ['cifar10','cifar100']
DIM=128

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

class CIFAR10_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class CIFAR10_DAE2(nn.Module):
    def __init__(self):
        super(CIFAR10_DAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,DIM,3,2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM,2*DIM,3,2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*DIM,4*DIM,3,2,padding=1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*DIM,2*DIM,2,stride=2),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*DIM,DIM,2,stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(DIM,3,2,stride=2),
            nn.Sigmoid(),
        )


    def forward(self,x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output.view(-1,3,32,32)

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
<<<<<<< HEAD
def cifar10(**kwargs):
    return CIFAR10(),CIFAR10_DAE()
=======
def cifar10():
    return CIFAR10(),CIFAR10_DAE2()
>>>>>>> dfc1a4b4b300b6850c23393982c08257cf1fc9b7

def cifar100():
    return None
