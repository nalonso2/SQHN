import VitMach
import math
import torch
from torch import nn
import torchvision
import numpy as np
import pickle
import utilities
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Subset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


#Reduces data to specified number of examples per category
def get_data(shuf=False, data=2):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if data == 0:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    elif data == 1:
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    elif data == 2:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    elif data == 3:
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform)
    else:
        dir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/train'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)
        dir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/test'
        testset = torchvision.datasets.ImageFolder(dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=shuf)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader, test_loader


def pretrn_online(arch=8, data=0, dev='cuda', max_iter=200, num_seeds=1, alpha=200, in_dim=64, in_chn=3):
    with torch.no_grad():
        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=in_dim, in_chnls=in_chn, wt_up=wtupType, alpha=alpha, arch=arch, chnls=chnls).to(dev)
            train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter)
            mem_images = torch.zeros(0, in_chn, in_dim, in_dim).to(dev)

            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)
                model.update_wts(images)


            torch.save(model, f'models/preTree_arch{arch}_data{data}_seed{s}')





pretrn_online(arch=6, data=4, dev='cuda', max_iter=2000, num_seeds=1, in_dim=64, in_chn=4, )
