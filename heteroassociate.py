import Tree
import Unit
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
def create_train_model(num_imgs, mod_type, data, dev='cuda', act=0):
    # Memorize
    train_loader = get_data(shuf=True, data=data, btch_size=num_imgs)
    for batch_idx, (images, y) in enumerate(train_loader):
        if mod_type == 0:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=0, actFunc=1).to(dev)
        elif mod_type == 1:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=4, actFunc=act).to(dev)
        elif mod_type == 2:
            model = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=act, arch=9, b_sim=4).to(dev)
        elif mod_type == 3:
            model = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=act, arch=8, b_sim=4).to(dev)
        elif mod_type == 4:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=5, actFunc=act).to(dev)
        elif mod_type == 5:
            model = MHN.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], lr=.03, beta=.00001*num_imgs, optim=1).to(dev)

        images = images.to(dev)
        if mod_type < 2 or mod_type > 3:
            model.load_wts(images.view(images.size(0), -1))
        else:
            model.load_wts(images)


        return model, images


def noise_test(n_seeds, noise, rec_thr, noise_tp=0, hip_sz=500, mod_type=0, data=0):

    rcll_acc = torch.zeros(n_seeds, len(noise)).to('cpu')
    rcll_mse = torch.zeros(n_seeds).to('cpu')
    for s in range(n_seeds):
        mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
        with torch.no_grad():
            mem_images = mem_images.to('cpu')

            for ns in range(len(noise)):
                if noise_tp == 0:
                    imgn = torch.clamp(mem_images + torch.randn_like(mem_images) * noise[ns], min=0.000001, max=1).to('cuda')
                else:
                    imgn = (mem_images + torch.randn_like(mem_images) * noise[ns]).to('cuda')


                #Recall and free up gpu memory
                p = mem_unit.recall(imgn).to('cpu')

                rcll_acc[s, ns] += ((torch.mean(torch.square((mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1))),
                                                 dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square(mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1)))

                p.to('cpu')
                imgn.to('cpu')
            mem_images.to('cpu')
            mem_unit.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)



def pixDrop_test(n_seeds, frc_msk, rec_thr, hip_sz=500, mod_type=0, data=0):

    rcll_acc = torch.zeros(n_seeds, len(frc_msk))
    rcll_mse = torch.zeros(n_seeds)
    for s in range(n_seeds):
        mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
        with torch.no_grad():
            mem_images = mem_images.to('cpu')

            for msk in range(len(frc_msk)):
                imgDrop = mem_images.clone().to('cuda')
                mask = torch.rand_like(mem_images[0]) > frc_msk[msk]
                mask = mask.repeat(mem_images.size(0),1,1,1).to('cuda')

                for c in range(imgDrop.size(1)):
                    imgDrop[:, c, :, :] *= mask[:,0,:,:]

                p = mem_unit.recall(imgDrop).to('cpu')
                mask = mask.to('cpu')
                p = p.view(p.size(0), 3, mem_images.size(2), mem_images.size(3))
                rcll_acc[s, msk] += ((torch.mean(torch.square(((mem_images - p) * mask).reshape(hip_sz, -1)),
                                    dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square((mem_images - p) * mask).reshape(hip_sz, -1))

                imgDrop.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)





def right_mask_test(n_seeds, frc_msk, rec_thr, hip_sz=500, mod_type=0, data=0):
    rcll_acc = torch.zeros(n_seeds, len(frc_msk))
    rcll_mse = torch.zeros(n_seeds)

    for s in range(n_seeds):
        mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
        with torch.no_grad():
            mem_images = mem_images.to('cpu')

            for msk in range(len(frc_msk)):
                imgMsk = mem_images.clone()
                imgMsk = imgMsk.to('cuda')
                imgMsk[:,:, :, int(mem_images.size(2) - mem_images.size(2) * frc_msk[msk]):] = 0.

                '''plt.imshow(imgMsk[0].to('cpu').permute(1,2,0))
                plt.show()'''

                #Perform recall free up gpu memory
                p = mem_unit.recall(imgMsk).to('cpu')

                p = p.view(p.size(0), 3, mem_images.size(2), mem_images.size(3))
                lng = int(mem_images.size(2) * (1 - frc_msk[msk]))
                rcll_acc[s, msk] += ((torch.mean(torch.square(((mem_images[:,:,:,0:lng] - p[:,:,:,0:lng])).reshape(hip_sz, -1)),
                                                 dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square((mem_images[:,:,:,0:lng] - p[:,:,:,0:lng])).reshape(hip_sz, -1))

                imgMsk.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)








#Test recall
def recall(n_seeds, frcmsk, noise, rec_thr, test_t, hp_sz, mod_type=0, data=0):
    #Pix drop
    if test_t == 0:
        return pixDrop_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)
    #Mask
    elif test_t == 1:
        return right_mask_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)
    #Noise
    elif test_t == 2:
        return noise_test(n_seeds, noise, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)




#Trains
def train(model_type=0, num_seeds=10, hip_sz=[1000], frcmsk=[0], noise=[.2], test_t=0, rec_thr=.001, data=2, act=0):
    #Recall
    acc_means, acc_stds, mse_means, mse_stds = recall(num_seeds, frcmsk=frcmsk, noise=noise, rec_thr=rec_thr,
                                                      test_t=test_t, hp_sz=hip_sz, mod_type=model_type, data=data)

    print(f'Data:{data}', f'Test:{test_t}', 'ModType:', model_type, '\nAcc:', acc_means, acc_stds,
          '\nMSE:', mse_means, mse_stds)


    with open(f'data/HeteroA_Model{model_type}_ActF{act}_Test{test_t}_numN{hip_sz}_frcMsk{frcmsk}_data{data}.data', 'wb') as filehandle:
        pickle.dump([acc_means, acc_stds, mse_means, mse_stds], filehandle)