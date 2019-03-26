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

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('--data',type=str,default='mnist', help='mnist, mnist_fashion')

# classifier
parser.add_argument('--load_clf',type=str,defaut=None, help = 'Specify the load path of classifiers. {None:train}')

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
