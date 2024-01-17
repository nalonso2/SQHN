import MHN_Tree
import MHN_EWC_Tree
import math
import torch
from torch import nn
import torchvision
import numpy as np
import pickle
import random
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





def train_online(arch=2, data=0, dev='cuda', max_iter=200, num_seeds=10, beta=10, lr=.1, shuf=True, t_fq=200,
                 in_dim=28, in_chn=1, chnls=200, run_test=True, save_md=False, optim=0, r=1):

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
        if optim < 2 or optim == 3:
            model = MHN_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, optim=optim).to(dev)
        else:
            model = MHN_EWC_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, r=r).to(dev)

        train_loader, test_loader = get_data(shuf=shuf, data=data, max_iter=max_iter)
        mem_images = torch.zeros(0, in_chn, in_dim, in_dim).to(dev)

        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.to(dev)
            mem_images = torch.cat((mem_images, images), dim=0)
            model.recall_learn(images.detach())

            if batch_idx % t_fq == 0 and run_test:
                with torch.no_grad():
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images)
                    tst_mse = test(model, batch_idx, test_loader)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk
                    test_mse[s, int(batch_idx / t_fq)] = tst_mse
                    '''print(batch_idx, f'Optim{optim} MSE:{torch.mean(recall_mse[s, int(batch_idx / t_fq)])}, '
                                     f'Acc:{torch.mean(recall_pcnt[s, int(batch_idx / t_fq)])} '
                                     f'Noise MSE:{torch.mean(recall_mse_n[s, int(batch_idx / t_fq)])}, '
                                     f'Noise Acc:{torch.mean(recall_pcnt_n[s, int(batch_idx / t_fq)])} ')'''

            elif batch_idx % t_fq == 0:
                print(batch_idx)

            if save_md:
                torch.save(model, f'models/preTreeMHN_arch{arch}_optim{optim}_chnls{in_chn}_data{data}_seed{s}')

            if batch_idx >= max_iter:
                break
    with torch.no_grad():
        print(f'Optim:{optim} MSE:{torch.mean(recall_mse[:, -1])}, Acc:{torch.mean(recall_pcnt[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse)}, Cuml Acc:{torch.mean(recall_pcnt)} '
              f'test_mse:{torch.mean(test_mse[:, -1])}')

    with open(f'data/AATreeBP_Online_arch{arch}_numN{chnls}_data{data}_numData{max_iter}_optim{optim}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk,
                         test_mse], filehandle)



def train_onCont(arch=3, data=0, dev='cuda', max_iter=200,  num_seeds=10, beta=1000, shuf=True, t_fq=200,
                 in_dim=28, in_chn=1, run_test=True, save_md=False, chnls=200, optim=0, lr=.001, r=1):

    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_n = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_mse_msk = torch.zeros(num_seeds, int(max_iter / t_fq))
        recall_pcnt_msk = torch.zeros(num_seeds, int(max_iter / t_fq))
        test_mse = torch.zeros(num_seeds, int(max_iter / t_fq))

    # Memorize
    for s in range(num_seeds):
        if optim < 2 or optim == 3:
            model = MHN_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, optim=optim).to(dev)
        else:
            model = MHN_EWC_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, r=r).to(dev)

        train_loader, test_loader = get_data(shuf=shuf, data=data, cont=True, max_iter=max_iter)

        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev)
            mem_images = torch.cat((mem_images, images), dim=0)
            model.recall_learn(images.detach())

            if batch_idx % t_fq == 0 and run_test:
                with torch.no_grad():
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images)
                    tst_mse = test(model, batch_idx, test_loader)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk
                    test_mse[s, int(batch_idx / t_fq)] = tst_mse
                    '''print(batch_idx, f'Seed:{s}  ', f'None:{round(mse, 4), round(pct, 4)}  ',
                          f'Noise:{round(msen, 4), round(pctn, 4)}  ',
                          f'Mask:{round(mse_msk, 4), round(pct_msk, 4)}')'''
            elif batch_idx % t_fq == 0:
                print(batch_idx)

        if save_md:
            torch.save(model, f'models/preTree_arch{arch}_wtUp{wtupType}_chnls{in_chn}_data{data}_seed{s}')

    with torch.no_grad():
        print(f'Optim:{optim} MSE:{torch.mean(recall_mse[:, -1])}, Acc:{torch.mean(recall_pcnt[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse)}, Cuml Acc:{torch.mean(recall_pcnt)} '
              f'test_mse:{torch.mean(test_mse[:, -1])}')

    with open(f'data/AATreeBP_OnlineCont_arch{arch}_numN{chnls}_data{data}_numData{max_iter}_optim{optim}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk,
                         test_mse], filehandle)





########################################## Online Domain Incremental ###########################################
def train_onContDom(in_dim, in_chn=3, arch=0, dev='cuda', max_iter=5000, num_seeds=5, beta=10, chnls=200, lr=.001,
                    t_fq=100, shuf=True, cont=True, optim=0, r=.01):
    with torch.no_grad():
        recall_mse = torch.zeros(num_seeds, int(max_iter/ t_fq))
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

    # Memorize
    for s in range(num_seeds):
        if optim < 2 or optim == 3:
            model = MHN_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, optim=optim).to(dev)
        else:
            model = MHN_EWC_Tree.Tree(in_dim=in_dim, in_chnls=in_chn, beta=beta, lr=lr, arch=arch, chnls=chnls, r=r).to(dev)

        train_loader = get_dom_train_loader(shuf=shuf, online=online, iter_dom=int(max_iter/4))
        for batch_idx, (images, y) in enumerate(train_loader):
            if batch_idx == 0:
                mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)

            images = images.to(dev)

            #Pixel switch halfway through each data set if doing continual, else flip pixel with .5 probability
            if cont and ((batch_idx > int(max_iter*.25) and batch_idx < int(max_iter*.5)) or batch_idx > int(max_iter*.75)):
                images = images * -.5 + 1
            elif online and (torch.rand(1) > .5).item():
                images = images * -.5 + 1
            else:
                images *= .5

            mem_images = torch.cat((mem_images, images), dim=0)
            model.recall_learn(images.detach())

            if batch_idx % t_fq == 0:
                with torch.no_grad():
                    mse, pct, msen, pctn, mse_msk, pct_msk = corrupt_test(.2, 0, model, mem_images, rec_thr=.005)
                    recall_mse[s, int(batch_idx / t_fq)] = mse
                    recall_pcnt[s, int(batch_idx / t_fq)] = pct
                    recall_mse_n[s, int(batch_idx / t_fq)] = msen
                    recall_pcnt_n[s, int(batch_idx / t_fq)] = pctn
                    recall_mse_msk[s, int(batch_idx / t_fq)] = mse_msk
                    recall_pcnt_msk[s, int(batch_idx / t_fq)] = pct_msk
                    '''print(batch_idx, f'Seed:{s}  ', f'None:{round(mse, 4), round(pct, 4)}  ',
                          f'Noise:{round(msen, 4), round(pctn, 4)}  ',
                          f'Mask:{round(mse_msk, 4), round(pct_msk, 4)}')'''


    with torch.no_grad():
        print(f'Optim:{optim} MSE:{torch.mean(recall_mse[:, -1])}, Acc:{torch.mean(recall_pcnt[:, -1])} ,'
              f'Cuml MSE:{torch.mean(recall_mse)}, Cuml Acc:{torch.mean(recall_pcnt)}')


    with open(f'data/AATreeBP_OnlineContDom_arch{arch}_numN{chnls}_numData{max_iter}_optim{optim}{cont_nm}.data', 'wb') as filehandle:
            pickle.dump([recall_mse, recall_pcnt, recall_mse_n, recall_pcnt_n, recall_mse_msk, recall_pcnt_msk], filehandle)


############################################################################################################
def corrupt_test(noise, noise_tp, mem_unit, mem_images, rec_thr=.01):
    with torch.no_grad():
        img = mem_images.clone()
        if noise_tp == 0:
            imgn = torch.clamp(img + torch.randn_like(img) * noise, min=0, max=1)
        else:
            imgn = img + torch.randn_like(img) * noise

        imgMsk = img.clone()
        imgMsk[:,:, :, 0:int(img.size(2)*.5)] = 0

        #None
        p = mem_unit.recall(img)
        mse = torch.mean(torch.square(img - p)).item()
        recalled = ((torch.mean(torch.square(img.view(img.size(0), -1) - p.view(p.size(0), -1)),
                                  dim=1) <= rec_thr).sum() / img.size(0)).item()
        #Noise
        pn = mem_unit.recall(imgn)
        mse_n = torch.mean(torch.square(img - pn)).item()
        recalled_n = ((torch.mean(torch.square(img.view(img.size(0), -1) - pn.view(pn.size(0), -1)), dim=1) <= rec_thr).sum() / img.size(0)).item()

        #Mask
        pmsk = mem_unit.recall(imgMsk)
        mse_msk = torch.mean(torch.square(img - pmsk)).item()
        recalled_msk = ((torch.mean(torch.square(img.view(img.size(0), -1) - pmsk.view(pmsk.size(0), -1)),
                                 dim=1) <= rec_thr).sum() / img.size(0)).item()

        #Free up memory
        imgMsk.to('cpu')
        imgn.to('cpu')
        img.to('cpu')

        return mse, recalled, mse_n, recalled_n, mse_msk, recalled_msk



def test(mem_unit, b_idx, test_loader, dev='cuda'):
    with torch.no_grad():
        #Get test mse
        test_mse = 0
        num_imgs = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.to(dev)
            '''imgMsk = images.clone()
            imgMsk[:,:, :, 0:int(images.size(2)*.5)] = 0'''

            pt = mem_unit.recall(images)
            test_mse += torch.mean(torch.square(images.view(images.size(0), -1) - pt.view(images.size(0), -1)), dim=1).sum().item()
            images.to('cpu')
            num_imgs += images.size(0)
        test_mse /= num_imgs

        '''if batch_idx % 1000 == 0:
            for x in range(1):
                b = 0#torch.randint(0, images.size(0), (1,)).item()
                f, axarr = plt.subplots(3)
                axarr[0].imshow(images[b].permute(1, 2, 0).to('cpu'))
                axarr[1].imshow(imgMsk[b].permute(1, 2, 0).to('cpu'))
                axarr[2].imshow(pt[b,0:3].permute(1, 2, 0).to('cpu'))
                f.suptitle(f'Iteration:{b_idx}')
                plt.show()'''

        return test_mse