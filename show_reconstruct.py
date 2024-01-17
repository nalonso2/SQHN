import Tree
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


def get_train_loader(train_set, num_cls=10, shuf=True, iter_cls=50, cont=False):
    with torch.no_grad():
        if cont:
            sorted_data = []
            for c in range(num_cls):
                labels = train_set.targets
                idx = (torch.tensor(labels)==c).nonzero().view(-1)
                if shuf:
                    idx = idx.view(-1)[torch.randperm(idx.size(0))]

                for d in range(iter_cls):
                    sorted_data.append(train_set[idx[d]])

            return torch.utils.data.DataLoader(sorted_data, batch_size=1, shuffle=False)
        else:
            return torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=shuf)


#Reduces data to specified number of examples per category
def get_data(shuf=False, data=2, cont=False, max_iter=50000):
    with torch.no_grad():
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if data == 0:
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
            num_cls = 10

        elif data == 1:
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
            num_cls = 10

        elif data == 2:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
            num_cls = 10

        elif data == 3:
            trainset = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform)
            num_cls = 10

        elif data == 4:
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
            num_cls = 100

        elif data == 5:
            dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/tiny-imagenet-200/train'
            trainset = torchvision.datasets.ImageFolder(dir, transform=transform)
            dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/tiny-imagenet-200/test'
            testset = torchvision.datasets.ImageFolder(dir, transform=transform)
            num_cls = 200

        elif data == 6:
            trainset = torchvision.datasets.EMNIST(root='./data', train=True, download=True, transform=transform, split='byclass')
            testset = torchvision.datasets.EMNIST(root='./data', train=False, download=True, transform=transform, split='byclass')
            num_cls = 62


        train_loader = get_train_loader(trainset, num_cls=num_cls, shuf=shuf, iter_cls=int(max_iter/num_cls), cont=cont)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

        return train_loader, test_loader





def test_reconstruct(arch=6, data=5, dev='cuda', max_iter=200, wtupType=0, in_chn=3, s=0):
    with torch.no_grad():
        _, test_loader = get_data(shuf=True, data=data, max_iter=max_iter)
        model = torch.load(f'models/preTree_arch{arch}_wtUp{wtupType}_chnls{in_chn}_data{data}_seed{s}')
        model.lmbd = .01
        print(model.lmbd)
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.to(dev)
            imgMsk = images.clone()
            imgMsk[:,:, :, 0:int(images.size(2)*.5)] = 0

            pt = model.recall(imgMsk)
            f, axarr = plt.subplots(3)
            if in_chn==3:
                axarr[0].imshow(images[0,0:3].permute(1, 2, 0).to('cpu'))
                axarr[1].imshow(imgMsk[0, 0:3].permute(1, 2, 0).to('cpu'))
                axarr[2].imshow(pt[0,0:3].permute(1, 2, 0).to('cpu'))
            else:
                axarr[0].imshow(images[0,0].t().to('cpu'), cmap='gray')
                axarr[1].imshow(imgMsk[0,0].t().to('cpu'), cmap='gray')
                axarr[2].imshow(pt[0,0].t().to('cpu'), cmap='gray')

            plt.show()

test_reconstruct(arch=3, data=6, max_iter=500, in_chn=1)