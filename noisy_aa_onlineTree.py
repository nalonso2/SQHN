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


def get_train_loader(train_set, num_cls=10, shuf=True, iter_cls=50, cont=False, cls_inc=False):
    if cont and cls_inc:
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

    return train_loader






############################################# Online I.I.D. ######################################################
def train_online(in_dim, arch, in_chn, chnls, data, dev='cuda', max_iter=200, num_seeds=10, alpha=10, shuf=True,
                 ns_type=0, num_up=5, plus=False, lr=.9):
    with torch.no_grad():
        t_fq = int(max_iter / 10)
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq)+1)

        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=in_dim, in_chnls=in_chn, wt_up=0, alpha=alpha, arch=arch, chnls=chnls, lr=lr).to(dev)
            train_loader = get_data(shuf=shuf, data=data, max_iter=max_iter)

            for batch_idx, (images, y) in enumerate(train_loader):
                if batch_idx == 0:
                    mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

                images = images.to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)

                for n in range(num_up):
                    if ns_type == 0:
                        imgn_new = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1)
                    else:
                        imgn_new = torch.bernoulli(images)

                    if n > 0:
                        imgn = imgn * .5 + imgn_new * .5
                    else:
                        imgn = imgn_new

                    #If first iteration or not doing IPHN+ then update activities, else do not update hidden activities
                    if n == 0 or not plus:
                        model.update_wts(imgn)
                    else:
                        act[0] = imgn
                        model.update_wts2(act)


                    #Reset activities after weight updates, if doing IPHN+
                    if n == 0 and plus:
                        act = model.ff_max(imgn)



                if  batch_idx  % t_fq == 0:
                    mse, pct, msen, pctn = corrupt_test(.2, ns_type, model, mem_images)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    print(batch_idx, f'Seed:{s}  ', f'None:{round(mse, 4), round(pct, 4)}  ',
                          f'Noise:{round(msen, 4), round(pctn, 4)}  ')


                if batch_idx == max_iter:
                    break

        print(f'Plus:{plus}  PropRcl:{torch.mean(recall_pcnt[:, -1])} '
              f'MSE:{torch.mean(recall_mse[:, -1])} '
              f'MSE(Noise):{torch.mean(recall_mse_n[:, -1])}')

    with open(f'data/AATree_NoisyOnline_numN{chnls}_data{data}_numData{max_iter}_NUp{num_up}_NsType{ns_type}_plus{plus}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n], filehandle)




###############################################################################################################################
def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01):
    with torch.no_grad():
        img = mem_images.clone()
        if noise_tp == 0:
            imgn = torch.clamp(img + torch.randn_like(img) * noise, min=0, max=1)
        else:
            imgn = torch.bernoulli(img)

        #None
        p = mem_unit.recall(img)
        mse = torch.mean(torch.square(img - p)).item()
        recalled = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                dim=1) <= rec_thr).sum() / img.size(0)).item()


        '''plt.imshow(p[0].permute(1, 2, 0).to('cpu'))
        plt.axis('off')
        plt.show()

        plt.imshow(img[0].permute(1, 2, 0).to('cpu'))
        plt.axis('off')
        plt.show()'''

        #Noise
        p = mem_unit.recall(imgn)
        mse_n = torch.mean(torch.square(img - p)).item()
        recalled_n = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)), dim=1) <= rec_thr).sum() / img.size(0)).item()

        return mse, recalled, mse_n, recalled_n