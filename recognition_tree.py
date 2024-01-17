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



def get_data(data=0, shuf=True):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if data == 0:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        in_dist = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        out_dist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        in_dist = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        out_dist = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        #out_dist = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=shuf)
    in_dist_loader = torch.utils.data.DataLoader(in_dist, batch_size=10000, shuffle=False)
    out_dist_loader = torch.utils.data.DataLoader(out_dist, batch_size=10000, shuffle=False)

    return train_loader, in_dist_loader, out_dist_loader


############################################################################################################################
def pre_train(model, data_loader, max_iter=1000, dev='cuda'):
    with torch.no_grad():
        print('PRETRAIN LOWER LAYERS')
        for batch_idx, (images, y) in enumerate(data_loader):
            images = images.to(dev)
            for l in range(len(model.layers) - 1):
                model.update_layer(images, l)

            if batch_idx % 200 == 0:
                print(f'{round(batch_idx / max_iter * 100,2)}% Complete')
            if batch_idx == max_iter:
                return



###################################################################################################################
def train_online(in_dim=32, in_chn=3, chnls=200, dev='cuda', max_iter=200, wtupType=0, num_seeds=10, alpha=100000,
                 t_fq=100, gamma=.999, data=1, shuf=True, arch=5, noise=0):

    with torch.no_grad():
        ns_label = ''
        if noise > 0:
            ns_label = f'_Noise({noise})'

        rec = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_in = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_out = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)

        rec_std = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_in_std = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_out_std = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)

        rec_acc = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_acc_in = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)
        rec_acc_out = torch.zeros(num_seeds, int(max_iter / t_fq) + 1)

        # Memorize
        for s in range(num_seeds):
            model = Tree.Tree(in_dim=in_dim, in_chnls=in_chn, wt_up=wtupType, alpha=alpha, arch=arch,
                              chnls=chnls, gamma=gamma).to(dev)

            train_loader, in_dist_loader, out_dist_loader = get_data(data=data, shuf=shuf)

            #Pretrain lower layers
            pre_train(model, train_loader)

            for batch_idx, (images, y) in enumerate(train_loader):
                images = images.to(dev)
                if batch_idx == 0:
                    mem_images = torch.zeros(0, images.size(1), images.size(2), images.size(3)).to(dev)
                mem_images = torch.cat((mem_images, images), dim=0)

                #Update memory node
                if arch == 5:
                    model.update_layer(images, 1)
                else:
                    model.update_layer(images, 2)

                if batch_idx % t_fq == 0:
                    avg_err, std_err, num_rec = train_recogn(model, mem_images, ns=noise)
                    in_avg_err, in_std_err, in_num_rec, ndt = in_dist_recogn(model, in_dist_loader, mem_images.size(0), ns=noise)
                    out_avg_err, out_std_err, out_num_rec, ndt_out = out_dist_recogn(model, out_dist_loader, mem_images.size(0), ns=noise)
                    rec[s, int(batch_idx / t_fq)] = avg_err.item()
                    rec_std[s, int(batch_idx / t_fq)] = std_err.item()
                    rec_in[s, int(batch_idx / t_fq)] = in_avg_err.item()
                    rec_in_std[s, int(batch_idx / t_fq)] = in_std_err.item()
                    rec_out[s, int(batch_idx / t_fq)] = out_avg_err.item()
                    rec_out_std[s, int(batch_idx / t_fq)] = out_std_err.item()
                    rec_acc[s, int(batch_idx / t_fq)] = num_rec / mem_images.size(0)
                    rec_acc_in[s, int(batch_idx / t_fq)] = in_num_rec / ndt
                    rec_acc_out[s, int(batch_idx / t_fq)] = out_num_rec / ndt_out

                    print(batch_idx, f'Seed:{s}  ', f'Train MSE:{round(avg_err.item(), 3)} '
                                                    f'  In Dist. MSE:{round(in_avg_err.item(), 3)} '
                                                    f'  Out Dist. MSE:{round(out_avg_err.item(), 3)}'
                                                    f'  Acc:{round(((num_rec + in_num_rec + out_num_rec) / (mem_images.size(0) + ndt + ndt_out)).item(), 4)}'
                                                    f'  Acc (train):{round(((num_rec) / (mem_images.size(0))).item(), 4)}'
                                                    f'  Acc (inDist):{round(((in_num_rec) / (ndt)).item(), 4)}'
                                                    f'  Acc (outDist):{round(((out_num_rec) / (ndt_out)).item(), 4)}')

                if batch_idx == max_iter:
                    break

    print(f'Acc: {(torch.mean(rec_acc[:, -1]) + torch.mean(rec_acc_in[:, -1]) + torch.mean(rec_acc_out[:, -1])) / 3} '
          f'Acc (train):{torch.mean(rec_acc[:, -1])}, Acc (inD):{torch.mean(rec_acc_in[:, -1])}, '
          f'Acc (outD):{torch.mean(rec_acc_out[:, -1])}')


    with open(f'data/RecognTree_arch{arch}_data{data}_numData{max_iter}{ns_label}.data', 'wb') as filehandle:
            pickle.dump([rec, rec_std, rec_in, rec_in_std, rec_out, rec_out_std, rec_acc, rec_acc_in, rec_acc_out], filehandle)



########################################################################################################################
def train_recogn(mem_unit, mem_images, dev='cuda', ns=0):
    with torch.no_grad():
        images = mem_images.to(dev)
        out = mem_unit.recall(images)
        avg_err = torch.mean(torch.square(images - out))
        std_err = torch.std(torch.mean(torch.square(images - out), dim=1))
        img_new = torch.clamp(images + torch.randn_like(images) * ns, min=0, max=1)
        num_rec = mem_unit.recognize(img_new).sum()
        return avg_err, std_err, num_rec



########################################################################################################################
def in_dist_recogn(mem_unit, test_loader, num_images, dev='cuda', ns=0):
    with torch.no_grad():
        #Get test mse
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images[0:num_images].to(dev)
            out = mem_unit.recall(images)
            avg_err = torch.mean(torch.square(images - out))
            std_err = torch.std(torch.mean(torch.square(images - out), dim=1))
            img_new = torch.clamp(images + torch.randn_like(images) * ns, min=0, max=1)
            num_rec = (False == mem_unit.recognize(img_new)).sum()
            break

        return avg_err, std_err, num_rec, images.size(0)


########################################################################################################################
def out_dist_recogn(mem_unit, out_test_loader, num_images, dev='cuda', ns=0):
    with torch.no_grad():
        # Get test mse
        for batch_idx, (images, y) in enumerate(out_test_loader):
            images = images[0:num_images].to(dev)
            if images.size(1) > 784:
                images = -1 * images[0:num_images].to(dev) + 1
            out = mem_unit.recall(images)
            avg_err = torch.mean(torch.square(images - out))
            std_err = torch.std(torch.mean(torch.square(images - out), dim=1))
            img_new = torch.clamp(images + torch.randn_like(images) * ns, min=0, max=1)
            num_rec = (False == mem_unit.recognize(img_new)).sum()
            break
        return avg_err, std_err, num_rec, images.size(0)