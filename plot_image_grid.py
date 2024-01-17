import Tree
import Hierarchy
import Unit
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
def get_data(shuf=True, data=2, btch_size=1):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if data == 0:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif data == 1:
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif data == 2:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    elif data == 3:
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform)
    elif data == 4:
        dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/tiny-imagenet-200/val'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.CenterCrop(128)])
        dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/CalTech256'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=btch_size, shuffle=shuf)

    return train_loader



#Plot Images
def plot_ex(imgs, name):
    grid = make_grid(imgs, nrow=5)
    imgGrid = torchvision.transforms.ToPILImage()(grid)
    plt.imshow(imgGrid)
    plt.axis('off')
    plt.title(name)
    plt.show()


#
def create_train_model(num_imgs, data, dev='cuda'):

    # Memorize
    train_loader = get_data(shuf=False, data=data, btch_size=num_imgs)
    for batch_idx, (images, y) in enumerate(train_loader):
        images = images.to(dev)

        #Get one layer models (auto and hetero associative version)
        L1 = [None, None]
        L1[0] = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=2, actFunc=0).to(dev)
        L1[1] = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=4, actFunc=0).to(dev)
        L1[0].load_wts(images.view(images.size(0), -1))
        L1[1].load_wts(images.view(images.size(0), -1))

        # Get two layer models (auto and hetero associative version)
        L2 = [None, None]
        L2[0] = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=0, arch=9).to(dev)
        L2[1] = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=0, arch=9, b_sim=4).to(dev)
        L2[0].load_wts(images)
        L2[1].load_wts(images)

        # Get three layer models (auto and hetero associative version)
        L3 = [None, None]
        L3[0] = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=0, arch=8).to(dev)
        L3[1] = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=0, arch=8, b_sim=4).to(dev)
        L3[0].load_wts(images)
        L3[1].load_wts(images)

        return [L1, L2, L3], images




def create_grid(n_seeds=5, noise=.2, frc_msk=.25, hip_sz=500, data=2):
    with torch.no_grad():
        for s in range(n_seeds):
            models, mem_images = create_train_model(hip_sz, data)
            mem_images = mem_images[0].reshape(1,3,32,32).to('cuda')

            #List of corrupted/masked images
            crpt_imgs = []
            crpt_imgs.append(mem_images.clone().reshape(1,3,32,32).to('cuda'))

            # Get Noised Image
            crpt_imgs.append(torch.clamp(mem_images + torch.randn_like(mem_images) * noise, min=0.000001, max=1).to('cuda'))

            #Get Occlusioned Images
            x_ln = torch.randint(low=int(mem_images.size(2) * frc_msk + 1), high=int(mem_images.size(2)), size=(1,)).item()
            y_ln = int(int(frc_msk * mem_images.size(2) * mem_images.size(2)) / (x_ln))
            x1 = torch.randint(low=0, high=int(mem_images.size(2) - x_ln), size=(1,)).item()
            y1 = torch.randint(low=0, high=int(mem_images.size(3) - y_ln), size=(1,)).item()

            crpt_imgs.append(mem_images.clone().to('cuda'))
            crpt_imgs[-1][:, :, x1:x1 + x_ln, y1:y1 + y_ln] = 0.

            crpt_imgs.append(mem_images.clone().to('cuda'))
            crpt_imgs[-1][:, 0, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
            crpt_imgs[-1][:, 1, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
            crpt_imgs[-1][:, 2, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()

            crpt_imgs.append(mem_images.clone().to('cuda'))
            crpt_imgs[-1][:, :, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand_like(mem_images[:, :, x1:x1 + x_ln, y1:y1 + y_ln])

            #Get Masked Images
            crpt_imgs.append(mem_images.clone().to('cuda'))
            mask = torch.rand_like(mem_images[0]) > frc_msk
            mask = mask.repeat(mem_images.size(0), 1, 1, 1).to('cuda')
            for c in range(mask.size(1)):
                crpt_imgs[-1][:, c, :, :] *= mask[:, 0, :, :]


            crpt_imgs.append(mem_images.clone().to('cuda'))
            crpt_imgs[-1][:, :, :, int(mem_images.size(2) - mem_images.size(2) * frc_msk):] = 0.


            #Generate recalled images
            num_img = len(crpt_imgs)
            for m in range(3):
                for im in range(num_img):
                    if im < num_img - 2:
                        crpt_imgs.append(models[m][0].recall(crpt_imgs[im]).reshape(1, 3, mem_images.size(2), mem_images.size(3)))
                    else:
                        crpt_imgs.append(models[m][1].recall(crpt_imgs[im]).reshape(1, 3, mem_images.size(2), mem_images.size(3)))


            g = [x.reshape(3, 32, 32) for x in crpt_imgs]
            grid = make_grid(g, nrow=7)
            imgGrid = torchvision.transforms.ToPILImage()(grid)
            plt.imshow(imgGrid)
            plt.axis('off')
            plt.show()


create_grid(noise=.2, frc_msk=.25, hip_sz=1024)
create_grid(noise=.8, frc_msk=.75, hip_sz=128)