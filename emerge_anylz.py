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


def get_train_loader(train_set, num_cls=10, shuf=True, iter_cls=50, cont=False, max_iter=2500):
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
            dta = [(train_set[x]) for x in range(max_iter+1)]
            return torch.utils.data.DataLoader(dta, batch_size=1, shuffle=shuf)




#
def get_dom_train_loader(shuf=True, online=False, iter_dom=750):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainset_F = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform)

    if shuf:
        idx = [x for x in range(int(2 * iter_dom))]
        random.shuffle(idx)
        sorted_data = [(trainset[x]) for x in idx]
        sorted_data = sorted_data + [(trainset_F[x]) for x in idx]
    else:
        sorted_data = [(trainset[x]) for x in range(int(2 * iter_dom))]
        sorted_data = sorted_data + [(trainset_F[x]) for x in range(int(2 * iter_dom))]

    return torch.utils.data.DataLoader(sorted_data, batch_size=1, shuffle=online)




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
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

        return train_loader, test_loader





def train_online(arch=2, data=0, dev='cuda', max_iter=200, wtupType=0, num_seeds=10, alpha=10, shuf=True, t_fq=200,
                 in_dim=28, in_chn=1, chnls=200):
    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        num_n = torch.zeros(num_seeds, 3, int(max_iter / t_fq)+1)
        avg_lr = torch.zeros(num_seeds, 3, int(max_iter / t_fq)+1)
        num_wts = torch.zeros(num_seeds, 3, int(max_iter / t_fq)+1)

        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=in_dim, in_chnls=in_chn, wt_up=wtupType, alpha=alpha, arch=arch, chnls=chnls).to(dev)
            train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter)
            mem_images = torch.zeros(0, in_chn, in_dim, in_dim).to(dev)

            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)
                model.update_wts(images)

                if batch_idx % t_fq == 0:
                    mse, pct = corrupt_test(.2, 0, model, mem_images)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    nrns = model.get_avg_num_nrns()
                    num_n[s, :, int(batch_idx / t_fq)] = torch.tensor(nrns)
                    lr = model.get_avg_lr()
                    avg_lr[s, :, int(batch_idx / t_fq)] = torch.tensor(lr)
                    nwts = model.get_num_nonzero_wts()
                    num_wts[s, :, int(batch_idx / t_fq)] = torch.tensor(nwts)

                    print('\n\n', f'Iter:{batch_idx}', f'\nMSE:{torch.mean(recall_mse[s, int(batch_idx / t_fq)])}, '
                                     f'\nAcc:{torch.mean(recall_pcnt[s, int(batch_idx / t_fq)])} '
                                     f'\nAvg Num Nrns:{num_n[s, :, int(batch_idx / t_fq)]}, '
                                     f'\nAvg Lr:{avg_lr[s, :, int(batch_idx / t_fq)]} '
                                     f'\nAvg Num Wts:{num_wts[s, :, int(batch_idx / t_fq)]} ')


                if batch_idx >= max_iter:
                    break

    with open(f'data/Emerge_arch{arch}_numN{chnls}_data{data}_numData{max_iter}_wtupType{wtupType}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, num_n, avg_lr, num_wts], filehandle)


############################################################################################################
def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01):
    with torch.no_grad():
        num_img = mem_images.size(0)
        num_batch = int(num_img / 500) + 1

        mse = 0
        recalled = 0
        mse_n = 0
        recalled_n = 0
        mse_msk = 0
        recalled_msk = 0
        for b in range(num_batch):
            img = mem_images[int(b*500):int(b*500) + 500].clone()

            # None
            p = mem_unit.recall(img)
            mse += torch.mean(torch.square(img.view(img.size(0), -1) - p.view(img.size(0), -1)), dim=1).sum().item()
            recalled += ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                    dim=1) <= rec_thr).sum()).item()


            # Free up memory
            img.to('cpu')

        mse /= mem_images.size(0)
        recalled /= mem_images.size(0)


        return mse, recalled


