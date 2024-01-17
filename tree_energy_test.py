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
    elif data == 4:
        dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/tiny-imagenet-200/train'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)
        dirt = 'C:/Users/nalon/Documents/PythonScripts/DataSets/tiny-imagenet-200/test'
        testset = torchvision.datasets.ImageFolder(dirt, transform=transform)
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.CenterCrop(128)])
        dir = 'C:/Users/nalon/Documents/PythonScripts/DataSets/CalTech256'
        trainset = torchvision.datasets.ImageFolder(dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=shuf)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)

    return train_loader, test_loader



def train_online(arch=2, data=4, dev='cuda', max_iter=5000, wtupType=0, num_seeds=10, alpha=10, in_dim=28, in_chn=1, chnls=200,
                 shuf=True, b_sim=2, simf=1):
    with torch.no_grad():
        tn_energies = []
        tst_energies = []
        for d in range(4):
            tn_energies.append(torch.zeros(num_seeds, 3))
            tst_energies.append(torch.zeros(num_seeds, 3))

        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=in_dim, in_chnls=in_chn, wt_up=wtupType, chnls=chnls, alpha=alpha, arch=arch,
                              simFunc=simf, b_sim=b_sim, lmbd=.5).to(dev)

            train_loader, test_loader = get_data(shuf=shuf, data=data)
            mem_images = torch.zeros(0, in_chn, in_dim, in_dim).to(dev)

            #Train Network
            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)
                model.update_wts(images)

                if batch_idx % 100 == 0:
                    print(batch_idx)

                if batch_idx >= max_iter:
                    break

            #Get Energy with different inference processes
            get_train_energies(model, mem_images, tn_energies, s)
            get_test_energies(model, test_loader, tst_energies, s)

            print(f'\nSeed:{s}\n'
                  f'Max:{tn_energies[0][s]}, {tst_energies[0][s]}\n '
                  f'ArgMax:{tn_energies[1][s][0:2]}, {tst_energies[1][s][0:2]}\n'
                  f'Softmax:{tn_energies[2][s][0:2]}, {tst_energies[2][s][0:2]}\n '
                  f'Rand:{tn_energies[3][s][0:2]}, {tst_energies[3][s][0:2]}')

    with open(f'data/AA_Online_TreeEnergy_Arch{arch}_max_iter{max_iter}.data', 'wb') as filehandle:
            pickle.dump([tn_energies, tst_energies], filehandle)




def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01, rec_type=0):
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
            if noise_tp == 0:
                imgn = torch.clamp(img + torch.randn_like(img) * noise, min=0, max=1)
            else:
                imgn = img + torch.randn_like(img) * noise

            # None
            if rec_type == 0:
                p = mem_unit.infer_max(img)[0]
            elif rec_type == 1:
                p = mem_unit.infer_argmax(img)[0]
            mse += torch.mean(torch.square(img.view(img.size(0), -1) - p.view(img.size(0), -1)), dim=1).sum().item()
            recalled += ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                    dim=1) <= rec_thr).sum()).item()


        mse /= mem_images.size(0)
        recalled /= mem_images.size(0)

        return mse, recalled


def get_train_energies(model, mem_images, engs, seed, dev='cuda'):
    with torch.no_grad():
        images = mem_images.clone()
        #Just FF
        a = model.ff_max(images)
        engs[0][seed,0] = model.compute_energy(images, a)
        a = model.ff_argmax(images)
        engs[1][seed, 0] = model.compute_energy(images, a)
        a = model.ff_softmax(images, beta=100)
        engs[2][seed, 0] = model.compute_energy(images, a)
        a = [torch.rand_like(a[x]) for x in range(len(a))]
        engs[3][seed, 0] = model.compute_energy(images, a)

        # FF + FB
        a = model.infer_max(images)
        engs[0][seed, 1] = model.compute_energy(images, a)
        a = model.infer_argmax(images)
        engs[1][seed, 1] = model.compute_energy(images, a)
        a = model.infer_softmax(images, beta=100)
        engs[2][seed, 1] = model.compute_energy(images, a)
        a = [torch.rand_like(a[x]) for x in range(len(a))]
        engs[3][seed, 1] = model.compute_energy(images, a)

        # 3x FF + FB
        a = model.infer_max_iter(images)
        engs[0][seed, 2] = model.compute_energy(images, a)


def get_test_energies(model, test_loader, engs, seed, dev='cuda'):
    with torch.no_grad():
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images[0:1000].to(dev)

            #Just FF
            a = model.ff_max(images)
            engs[0][seed,0] = model.compute_energy(images, a)
            a = model.ff_argmax(images)
            engs[1][seed, 0] = model.compute_energy(images, a)
            a = model.ff_softmax(images, beta=100)
            engs[2][seed, 0] = model.compute_energy(images, a)
            a = [torch.rand_like(a[x]) for x in range(len(a))]
            engs[3][seed, 0] = model.compute_energy(images, a)

            # FF + FB
            a = model.infer_max(images)
            engs[0][seed, 1] = model.compute_energy(images, a)
            a = model.infer_argmax(images)
            engs[1][seed, 1] = model.compute_energy(images, a)
            a = model.infer_softmax(images, beta=100)
            engs[2][seed, 1] = model.compute_energy(images, a)
            a = [torch.rand_like(a[x]) for x in range(len(a))]
            engs[3][seed, 1] = model.compute_energy(images, a)

            #3x FF + FB
            a = model.infer_max_iter(images)
            engs[0][seed, 2] = model.compute_energy(images, a)

            break


#train_online(arch=2, data=4, dev='cuda', max_iter=200, wtupType=0, num_seeds=5, alpha=10000000, in_dim=64, in_chn=3, shuf=True)
train_online(arch=2, data=4, dev='cuda', max_iter=1000, wtupType=0, num_seeds=3, alpha=10000000, in_dim=64, in_chn=3, shuf=True)