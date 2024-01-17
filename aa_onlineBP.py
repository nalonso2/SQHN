import MHN
import MHN_EWC
import math
import torch
from torch import nn
import torchvision
import numpy as np
import random
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



def get_dom_train_loader(shuf=True, online=False, iter_dom=750):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    trainset_F = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)

    if shuf:
        idx = [x for x in range(int(2 * iter_dom))]
        random.shuffle(idx)
        sorted_data = [(trainset[x]) for x in idx]
        sorted_data = sorted_data + [(trainset_F[x]) for x in idx]
    else:
        sorted_data = [(trainset[x]) for x in range(int(2*iter_dom))]
        sorted_data = sorted_data + [(trainset_F[x]) for x in range(int(2*iter_dom))]


    return torch.utils.data.DataLoader(sorted_data, batch_size=1, shuffle=online)



#Reduces data to specified number of examples per category
def get_data(shuf=False, data=2, cont=False, max_iter=50000):
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


    train_loader = get_train_loader(trainset, num_cls=num_cls, shuf=shuf, iter_cls=int(max_iter/num_cls), cont=cont)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader, test_loader






############################################################################################################################
def train_online(in_sz, hid_sz, data, dev='cuda', max_iter=3000, num_seeds=10, beta=10, lr=.2, t_fq=100, shuf=True,
                 opt=0, r=10):
    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_mse_msk = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt_msk = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        test_mse = torch.zeros(num_seeds, int(max_iter / t_fq)+1)


    # Memorize
    for s in range(num_seeds):
        if opt < 2 or opt == 3:
            model = MHN.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, optim=opt).to(dev)
        else:
            model = MHN_EWC.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, r=r).to(dev)

        train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter, cont=False)

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev) * .99 + .005
            mem_images = torch.cat((mem_images, images), dim=0)
            images = images.view(1, -1)

            model.recall_learn(images)
            with torch.no_grad():
                if batch_idx % t_fq == 0:
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images)
                    tst_mse = test(model, test_loader, dev)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk
                    test_mse[s, int(batch_idx / t_fq)] = tst_mse
                    '''print(batch_idx, f'Seed:{s}  ', f'None:{round(mse, 4), round(pct, 4)}  ',
                          f'Noise:{round(msen, 4), round(pctn, 4)}  ',
                          f'Mask:{round(mse_msk, 4), round(pct_msk, 4)}  ',
                          f'Test MSE:{round(tst_mse, 4)}')'''


            if batch_idx == max_iter:
                break

    with torch.no_grad():
        print(f'Optim:{opt} MSE:{torch.mean(recall_mse_n[:, -1])}, Acc:{torch.mean(recall_pcnt_n[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse_n)}, Cuml Acc:{torch.mean(recall_pcnt_n)}')

    with open(f'data/AA_OnlineBP_data{data}_opt{opt}_hdsz{hid_sz}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk,
                         test_mse, in_sz*hid_sz], filehandle)







###################################################################################################################
def train_onCont(in_sz, hid_sz, data, dev='cuda', max_iter=5000, num_seeds=10, beta=10, lr=.2, t_fq=100, shuf=True,
                 opt=0, r=10):
    with torch.no_grad():
        if data < 4:
            num_cls = 10
        elif data == 4:
            num_cls = 100
        elif data == 5:
            num_cls = 200
        elif data == 6:
            num_cls = 62

        recall_mse = torch.zeros(num_seeds, int(max_iter/ t_fq))
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_msk = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_msk = torch.zeros(num_seeds, int(max_iter / t_fq))
        test_mse = torch.zeros(num_seeds, int(max_iter / t_fq))

    # Memorize
    for s in range(num_seeds):
        if opt < 2 or opt == 3:
            model = MHN.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, optim=opt).to(dev)
        else:
            model = MHN_EWC.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, r=r).to(dev)
        train_loader, test_loader = get_data(shuf=shuf, data=data, cont=True, max_iter=max_iter)

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev) * .99 + .005
            mem_images = torch.cat((mem_images, images), dim=0)
            images = images.view(1, -1)

            model.recall_learn(images)

            with torch.no_grad():
                if batch_idx % t_fq == 0:
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images)
                    tst_mse = test(model, test_loader, dev)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk
                    test_mse[s, int(batch_idx / t_fq)] = tst_mse

    with torch.no_grad():
        print(f'Optim:{opt} MSE:{torch.mean(recall_mse_n[:, -1])}, Acc:{torch.mean(recall_pcnt_n[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse_n)}, Cuml Acc:{torch.mean(recall_pcnt_n)}')


    with open(f'data/AA_OnlineContBP_data{data}_opt{opt}_hdsz{hid_sz}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk,
                         test_mse], filehandle)





############################################################################################################################
def train_onContDom(in_sz, hid_sz, dev='cuda', max_iter=200, num_seeds=10, beta=10, lr=.2, t_fq=100, shuf=True,
                    opt=0, cont=True, r=10):
    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_msk = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_msk = torch.zeros(num_seeds, int(max_iter / t_fq))

        cont_nm = ''
        online = False
        if not cont:
            cont_nm = 'online'
            online = True


    for s in range(num_seeds):
        if opt < 2 or opt == 3:
            model = MHN.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, optim=opt).to(dev)
        else:
            model = MHN_EWC.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, r=r).to(dev)

        train_loader = get_dom_train_loader(shuf=shuf, online=online, iter_dom=int(max_iter / 4))

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev) * .99 + .005

            # Pixel switch halfway through each dataset if doing continual, else flip pixel with .5 probability
            if cont and ((batch_idx > int(max_iter * .25) and batch_idx < int(max_iter * .5)) or batch_idx > int(
                    max_iter * .75)):
                images = (images - 1) * -1
            elif online and (torch.rand(1) > .5).item():
                images = (images - 1) * -1

            mem_images = torch.cat((mem_images, images), dim=0)
            images = images.view(1, -1)

            model.recall_learn(images)


            with torch.no_grad():
                if batch_idx % t_fq == 0:
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk


    with torch.no_grad():
        print(f'Optim:{opt} MSE:{torch.mean(recall_mse_n[:, -1])}, Acc:{torch.mean(recall_pcnt_n[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse_n)}, Cuml Acc:{torch.mean(recall_pcnt_n)}')

    with open(f'data/AA_OnlineContDomBP_opt{opt}_hdsz{hid_sz}{cont_nm}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk,
                         in_sz*hid_sz], filehandle)



###############################################################################################################################
def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01):
    with torch.no_grad():
        img = mem_images.clone()
        if noise_tp == 0:
            imgn = torch.clamp(img + torch.randn_like(img) * noise, min=0, max=1).view(img.size(0), -1)
        else:
            imgn = (img + torch.randn_like(img) * noise).view(img.size(0), -1)

        imgMsk = img.clone()
        imgMsk[:,:, 0:int(img.size(2) * .5), :] = 0
        imgMsk = imgMsk.view(img.size(0), -1)

        #None
        p = mem_unit.recall(img.view(img.size(0), -1))
        mse = torch.mean(torch.square(img.view(img.size(0), -1) - p)).item()
        recalled = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                  dim=1) <= rec_thr).sum() / img.size(0)).item()

        #Noise
        p = mem_unit.recall(imgn)
        mse_n = torch.mean(torch.square(img.view(img.size(0), -1) - p)).item()
        recalled_n = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)), dim=1) <= rec_thr).sum() / img.size(0)).item()


        #Mask
        p = mem_unit.recall(imgMsk)
        mse_msk = torch.mean(torch.square(img.view(img.size(0), -1) - p)).item()
        recalled_msk = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                 dim=1) <= rec_thr).sum() / img.size(0)).item()


        return mse, recalled, mse_n, recalled_n, mse_msk, recalled_msk




def test(mem_unit, test_loader, dev='cuda'):
    with torch.no_grad():
        #Get test mse
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(dev)
            pt = mem_unit.recall(images)
            test_mse = torch.mean(torch.square(images - pt)).item()
            break

        return test_mse