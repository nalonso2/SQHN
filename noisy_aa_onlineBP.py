import MHN
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
def get_data(shuf=False, data=2, cont=False, max_iter=50000, iter_cls=50):
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
        dir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/train'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)
        dir = 'C:/Users/nalon/Documents/PythonScripts/tiny-imagenet-200/test'
        testset = torchvision.datasets.ImageFolder(dir, transform=transform)
        num_cls = 200

    elif data == 6:
        trainset = torchvision.datasets.EMNIST(root='./data', train=True, download=True, transform=transform, split='byclass')
        testset = torchvision.datasets.EMNIST(root='./data', train=False, download=True, transform=transform, split='byclass')
        num_cls = 62


    train_loader = get_train_loader(trainset, num_cls=num_cls, shuf=shuf, iter_cls=iter_cls, cont=cont)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader, test_loader






############################################################################################################################
def train_online(in_sz, hid_sz, data, dev='cuda', max_iter=200, num_seeds=10, beta=10, lr=.2, t_fq=100, shuf=True,
                 opt=0, num_up=1, ns_type=0):
    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)

    # Memorize
    for s in range(num_seeds):
        model = MHN.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, optim=opt).to(dev)
        train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter, cont=False)

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev) * .96 + .02
            mem_images = torch.cat((mem_images, images), dim=0)
            images = images.view(1, -1)

            for n in range(num_up):
                with torch.no_grad():
                    if ns_type == 0:
                        imgn = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1)
                        t_new = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1)
                    else:
                        imgn = torch.bernoulli(images)
                        t_new = torch.bernoulli(images)

                    if n > 0:
                        t = t * .5 + t_new * .5
                    else:
                        t = t_new

                model.recall_learn(imgn.detach(), t.detach())


            with torch.no_grad():
                if batch_idx % t_fq == 0:
                    mse, pct, msen, pctn = corrupt_test(.2, ns_type, model, mem_images)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    '''print(batch_idx, f'Seed:{s}  ', f'None:{round(mse, 4), round(pct, 4)}  ',
                          f'Noise:{round(msen, 4), round(pctn, 4)}  ')'''


            if batch_idx == max_iter:
                break

    print(f'PropRcl:{torch.mean(recall_pcnt[:, -1])} '
          f'MSE:{torch.mean(recall_mse[:, -1])} '
          f'MSE(Noise):{torch.mean(recall_mse_n[:, -1])}')

    with open(f'data/AA_NoisyOnlineBP_data{data}_opt{opt}_NUp{num_up}_NsType{ns_type}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n], filehandle)




###############################################################################################################################
def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01):
    with torch.no_grad():
        img = mem_images.clone()
        if noise_tp == 0:
            imgn = torch.clamp(img + torch.randn_like(img) * noise, min=0, max=1).view(img.size(0), -1)
        else:
            imgn = torch.bernoulli(img).view(img.size(0), -1)

        #None
        p = mem_unit.recall(img.view(img.size(0), -1))
        mse = torch.mean(torch.square(img.view(img.size(0), -1) - p)).item()
        recalled = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                  dim=1) <= rec_thr).sum() / img.size(0)).item()

        #Noise
        p = mem_unit.recall(imgn)
        mse_n = torch.mean(torch.square(img.view(img.size(0), -1) - p)).item()
        recalled_n = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)), dim=1) <= rec_thr).sum() / img.size(0)).item()

        return mse, recalled, mse_n, recalled_n