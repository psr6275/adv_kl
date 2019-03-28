import matplotlib.pyplot as plt
import torch
import torch.nn
import torchvision
import numpy as np

def show_prob(y_tup,y_true):
    """

    :param y_tup: we want to get numpy tuple
    :param y_true: we want to get numpy array instead of tensor
    :return:
    """
    num_comp = len(y_tup)
    #assert num_comp<=4 and num_comp>1, "Error! the number of comparison should be between 1 and 4"
    num_inst,num_class = y_tup[0].shape
    for i in range(num_inst):
        for j in range(num_comp):
            plt.subplot(1,num_comp,j+1)
            plt.bar(np.arange(num_class),y_tup[j][i],align='center',bar_width=0.8)
        plt.title(y_true[i])
        plt.show()

def show_image(X_tup,tup_name = None, channel_first = True):
    """

    :param X_tup: should be numpy array.
    :param channel_first: (batch, channel, width,height)
    :return:
    """
    num_comp = len(X_tup)
    num_inst = X_tup[0].shape[0]
    if channel_first:
        for i in range(num_comp):
            X_tup[i] = X_tup[i].transpose([0,2,3,1])
    for i in range(num_inst):
        for j in range(num_comp):
            plt.subplot(1,num_comp,j+1)
            plt.imshow(X_tup[j][i])
            if tup_name is not None:
                plt.title(tup_name[j])
            plt.axis('off')
        plt.show()
