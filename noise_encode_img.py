import Unit
import Tree
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
import seaborn as sns
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
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    return train_loader, test_loader






############################################# Online I.I.D. ######################################################
def train_online(in_sz, hid_sz, simf, data, dev='cuda', max_iter=200, wtupType=1, num_seeds=10, alpha=10, lr=.2,
                 shuf=False, det_type=0, ns_type=0, num_up=17, plus=False):

    with torch.no_grad():
        # Memorize
        for s in range(num_seeds):
            model = Unit.MemUnit(layer_szs=[in_sz, hid_sz], simFunc=simf, wt_up=wtupType, alpha=alpha, lr=lr,
                                 det_type=det_type).to(dev)
            train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter, cont=False)

            for batch_idx, (images, y) in enumerate(train_loader):
                if batch_idx == 0:
                    mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

                images = images.to(dev) * .96 + .02
                mem_images = torch.cat((mem_images, images), dim=0)
                images = images.view(1, -1)

                inp = []
                outp = []
                for n in range(num_up):
                    if ns_type == 0:
                        imgn_new = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1)
                    else:
                        imgn_new = torch.bernoulli(images)

                    if n > 0:
                        imgn = imgn_new
                    else:
                        imgn = imgn_new

                    # If first iteration or not doing SQHN+ then update activities
                    if n == 0 or not plus:
                        act = model.infer_step(imgn)

                    z = F.one_hot(torch.argmax(act, dim=1), num_classes=model.layer_szs[1]).float().to(dev)
                    model.update_wts(act, z, imgn)

                    # Reset activities after weight updates, if doing SQHN+
                    if n == 0 and plus:
                        act = model.infer_step(imgn)

                    if n % 6 == 0:
                        inp.append(imgn.clone())
                        outp.append(model.wts(z))


                #Show learning from noisy input
                f, axarr = plt.subplots(2,len(inp), figsize=(5, 5))
                axarr[0,0].set(ylabel=f'Noisy Input')
                axarr[1,0].set(ylabel=f'SQHN+\nOutput')

                for x in range(len(inp)):
                    axarr[0,x].set(title=f'Iter {x * 6}')
                    axarr[0,x].imshow(inp[x].reshape(28,28).to('cpu'))
                    axarr[0,x].tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False)
                    axarr[1,x].imshow(outp[x].reshape(28,28).to('cpu'))
                    axarr[1, x].tick_params(left=False, right=False, labelleft=False,
                                            labelbottom=False, bottom=False)

                #f.suptitle(f'SQHN', fontsize=15)
                #f.tight_layout(pad=.1)
                plt.subplots_adjust(wspace=0.01,
                                    hspace=0.01)
                plt.show()


                # Show Original
                f, axarr = plt.subplots(1)
                axarr.set(title=f'Original')
                axarr.imshow(images.reshape(28, 28).to('cpu'))
                axarr.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                plt.show()





############################################# Online Tree ######################################################
def train_tree(data, dev='cuda', max_iter=200, num_seeds=10, shuf=False, ns_type=0, num_up=13, plus=False):

    with torch.no_grad():
        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=32, in_chnls=3, wt_up=0, alpha=100000, arch=2, chnls=200, lr=1.).to(dev)
            train_loader,_ = get_data(shuf=shuf, data=data, max_iter=max_iter)

            for batch_idx, (images, y) in enumerate(train_loader):
                if batch_idx == 0:
                    mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

                images = images.to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)

                inp = []
                outp = []
                for n in range(num_up):
                    if ns_type == 0:
                        imgn = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1)
                    else:
                        imgn = torch.bernoulli(images)


                    # If first iteration or not doing SQHN+ then update activities, else do not update hidden activities
                    if n == 0 or not plus:
                        model.update_wts(imgn)
                    else:
                        act[0] = imgn
                        model.update_wts2(act)

                    # Reset activities after weight updates, if doing SQHN+
                    if n == 0 and plus:
                        act = model.ff_max(imgn)

                    if n % 6 == 0:
                        inp.append(imgn.clone())
                        outp.append(model.recall(images))


                #Show learning from noisy input
                f, axarr = plt.subplots(2,len(inp), figsize=(5, 5))
                axarr[0,0].set(ylabel=f'Noisy Input')
                axarr[1,0].set(ylabel=f'SQHN+\nOutput')

                for x in range(len(inp)):
                    axarr[0,x].set(title=f'Iter {x * 6}')
                    print(inp[x].shape)
                    axarr[0,x].imshow(inp[x][0].permute(1,2,0).to('cpu'))
                    axarr[0,x].tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False)

                    axarr[1,x].imshow(outp[x][0].permute(1,2,0).to('cpu'))
                    axarr[1, x].tick_params(left=False, right=False, labelleft=False,
                                            labelbottom=False, bottom=False)

                #f.suptitle(f'SQHN', fontsize=15)
                #f.tight_layout(pad=.1)
                plt.subplots_adjust(wspace=0.01,
                                    hspace=0.01)
                plt.show()


                # Show Original
                f, axarr = plt.subplots(1)
                axarr.set(title=f'Original')
                axarr.imshow(images[0].permute(1,2,0).to('cpu'))
                axarr.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                plt.show()




################################################ Online I.I.D. MHN #####################################################
def train_online_MHN(in_sz, hid_sz, data, dev='cuda', max_iter=200, num_seeds=10, beta=10, lr=.2, t_fq=100, shuf=False,
                 opt=0, num_up=10, ns_type=0):

    # Memorize
    for s in range(num_seeds):
        model = MHN.MemUnit(layer_szs=[in_sz, hid_sz], beta=beta, lr=lr, optim=opt).to(dev)
        train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter, cont=False)

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev) * .99 + .005
            mem_images = torch.cat((mem_images, images), dim=0)
            images = images.view(1, -1)

            inp = []
            outp = []
            for n in range(num_up):
                with torch.no_grad():
                    if ns_type == 0:
                        imgn = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1).view(images.size(0), -1)
                        t = torch.clamp(images + torch.randn_like(images) * .2, min=0, max=1).view(images.size(0), -1)
                    else:
                        imgn = torch.bernoulli(images * .9 + .05).view(images.size(0), -1)
                        t = torch.bernoulli(images * .9 + .05).view(images.size(0), -1)

                model.recall_learn(imgn.detach(), t.detach())

                if n % 2 == 0:
                    with torch.no_grad():
                        inp.append(imgn.clone())
                        outp.append(model.recall(imgn))

            # Show learning from noisy input
            f, axarr = plt.subplots(2, len(inp), figsize=(5, 5))
            axarr[0, 0].set(ylabel=f'Noisy Input')
            axarr[1, 0].set(ylabel=f'Output')

            for x in range(len(inp)):
                axarr[0, x].set(title=f'Iter {x * 2 + 1}')
                axarr[0, x].imshow(inp[x].permute(1,2,0).to('cpu'))
                axarr[0, x].tick_params(left=False, right=False, labelleft=False,
                                        labelbottom=False, bottom=False)
                axarr[1, x].imshow(outp[x].permute(1,2,0).to('cpu'))
                axarr[1, x].tick_params(left=False, right=False, labelleft=False,
                                        labelbottom=False, bottom=False)

            f.suptitle(f'MHN', fontsize=15)
            # f.tight_layout(pad=.1)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.show()


#Bernoulli
train_online(784, 300, simf=2, data=0, dev='cuda', max_iter=300, wtupType=0, alpha=1000000, det_type=3,
             num_seeds=1, lr=1., ns_type=1, plus=True)

#train_tree(data=4, dev='cuda', max_iter=300, num_seeds=1, ns_type=1, plus=True)
