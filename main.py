import argparse
import os
import numpy as np
import time, pickle,sys, json, seaborn, PIL, tempfile, warnings

import torch
import torch.nn as nn
import torch.optm as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as data

import matplotlib.pyplot as plt

# User-defined parts
from utils import mkdir_p
import models

model_namse = sorted(name for name in models.__dict__)
parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--data',type=str,default='mnist', help='mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--workers', default=16, type=int, metavar='N',help='number of data loading workers (default:16)')

# classifier
parser.add_argument('--load_clf',type=str,defaut=None, help = 'Specify the load path of classifiers. {None:train}')
parser.add_argument('--tansform',type=bool,default=False,help='add transformation for (clf or dae)')

# PGD attacks
parser.add_argument('--eps',type=float,default=8/255. help='pgd bound')
parser.add_argument('--niters',type=int,default=20, help='pgd steps')
parser.add_argument('--alpha',type=float,default=2/255,help='pgd steps')

# DAE part
parser.add_argument('--dae_loss',type=str,default='KL',help='KL, L2, L1')

# Checkpoints
parser.add_argument('-c','--checkpoint',default='checkpoint',type=str,metavar='PATH',
                    help='path to save checkpoint {default:checkpoint}')

args = parser.parse_args()

def main():
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print('==> preparing dataset %s',args.dataset)

    if args.dataset =='mnist':
        dataloader = datasets.MNIST
    elif args.dataset =='fashion_mnist':
        dataloader = datasets.FashionMNIST
    elif args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    elif args.dataset =='cifar100':
        dataloader = datasets.CIFAR100

    if args.transform and args.dataset in ['cifar10','cifar100']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])



    trainset = dataloader(root = './data',train=True, download=True,transform = transform_train)
    trainloader = data.Dataloader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False,transform=transform_test)
    testloader = data.Dataloader(testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    ## classifier model





