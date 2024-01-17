from torch import nn
import torchvision
import numpy as np
import pickle
import utilities
from torch.utils.data import DataLoader
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Subset



#Reduces data to specified number of examples per category
def get_data(shuf=False, data=2, cont=False, max_iter=50000, iter_cls=50):
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
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            num_cls = 10

        elif data == 3:
            trainset = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform)
            return torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

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

        return train_loader


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


def plot_image(data=0, cont=True, flip=False):
    with torch.no_grad():
        train_loader = get_data(shuf=False, data=data, max_iter=2000, cont=cont)

        for batch_idx, (images, y) in enumerate(train_loader):
            if (data == 0 or data == 6 or data == 1) and not flip:
                plt.imshow(images.view(28, 28), cmap='gray')
                plt.axis('off')
                plt.show()
            elif (data == 0 or data == 6 or data == 1) and flip:
                print('here')
                images = images * -1 + 1
                plt.imshow(images.view(28, 28), cmap='gray')
                plt.axis('off')
                plt.show()
            elif not flip:
                images = images
                plt.imshow(images[0].permute(1, 2, 0).to('cpu'))
                plt.axis('off')
                plt.show()
            elif flip:
                images = images * -.5 + 1
                plt.imshow(images[0].permute(1, 2, 0).to('cpu'))
                plt.axis('off')
                plt.show()





def plot_mask_image(data=5, frc_msk=.3):
    with torch.no_grad():
        train_loader = get_data(shuf=False, data=data, max_iter=200, cont=False)

        for batch_idx, (images, y) in enumerate(train_loader):
            imgMsk = images.clone()

            x_ln = torch.randint(low=int(imgMsk.size(2) * frc_msk + 1), high=int(imgMsk.size(2)), size=(1,)).item()
            y_ln = int(int(frc_msk * imgMsk.size(2) * imgMsk.size(2)) / (x_ln))

            x1 = torch.randint(low=0, high=int(imgMsk.size(2) - x_ln), size=(1,)).item()
            y1 = torch.randint(low=0, high=int(imgMsk.size(3) - y_ln), size=(1,)).item()

            imgMsk[:, :, x1:x1 + x_ln, y1:y1 + y_ln] = 0.

            imgClrMsk = images.clone()
            imgClrMsk[:, 0, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
            imgClrMsk[:, 1, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
            imgClrMsk[:, 2, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()

            imgNsMsk = images.clone()
            imgNsMsk[:, :, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand_like(imgMsk[:, :, x1:x1 + x_ln, y1:y1 + y_ln])

            plt.imshow(imgClrMsk[0].permute(1,2,0))
            plt.axis('off')
            plt.show()

            plt.imshow(imgMsk[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()

            plt.imshow(imgNsMsk[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()

            plt.imshow(images[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()



def plot_image_noise(data=4):
    with torch.no_grad():
        train_loader = get_data(shuf=False, data=data, max_iter=200, cont=False)

        for batch_idx, (images, y) in enumerate(train_loader):

            images = torch.clamp(images + .1 * torch.randn_like(images), min=0., max=1.)

            plt.imshow(images[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()



plot_image_noise()