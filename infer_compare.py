import Tree
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
def create_train_model(num_imgs, mod_type, data, dev='cuda', act=0):
    # Memorize
    train_loader = get_data(shuf=True, data=data, btch_size=num_imgs)
    for batch_idx, (images, y) in enumerate(train_loader):
        if mod_type == 0:
            model = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=act, arch=7).to(dev)
        elif mod_type == 1:
            model = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=act, arch=8).to(dev)
        elif mod_type == 2:
            model = Tree.Tree(in_dim=images.size(2), chnls=num_imgs, actFunc=act, arch=9).to(dev)


        images = images.to(dev)
        model.load_wts(images)

        return model, images




def noise_test(n_seeds, noise, rec_thr, noise_tp=0, hip_sz=500, mod_type=0, data=0, infer_type=0):
    with torch.no_grad():
        rcll_acc = torch.zeros(n_seeds, len(noise))
        rcll_mse = torch.zeros(n_seeds, len(noise))
        for s in range(n_seeds):
            mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
            mem_images = mem_images.to('cpu')

            for ns in range(len(noise)):
                if noise_tp == 0:
                    imgn = torch.clamp(mem_images + torch.randn_like(mem_images) * noise[ns], min=0.000001, max=1).to('cuda')
                else:
                    imgn = (mem_images + torch.randn_like(mem_images) * noise[ns]).to('cuda')


                #Recall and free up gpu memory
                if infer_type == 0:
                    p = mem_unit.infer_max(imgn)[0].to('cpu')
                elif infer_type == 1:
                    p = mem_unit.infer_argmax(imgn)[0].to('cpu')
                elif infer_type == 2:
                    p = mem_unit.infer_softmax(imgn)[0].to('cpu')

                rcll_acc[s, ns] += ((torch.mean(torch.square((mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1))),
                                                 dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s,ns] += torch.mean(torch.square(mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1)))

                imgn.to('cpu')
                p.to('cpu')
            mem_unit.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)



def mask_test(n_seeds, frc_msk, rec_thr, hip_sz=500, mod_type=0, data=0, msk_typ=0, infer_type=0):
    with torch.no_grad():
        rcll_acc = torch.zeros(n_seeds, len(frc_msk))
        rcll_mse = torch.zeros(n_seeds, len(frc_msk))

        for s in range(n_seeds):
            mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
            mem_images = mem_images.to('cpu')

            for msk in range(len(frc_msk)):
                imgMsk = mem_images.clone()
                imgMsk = imgMsk.to('cpu')

                x_ln = torch.randint(low=int(imgMsk.size(2)*frc_msk[msk]+1), high=int(imgMsk.size(2)), size=(1,)).item()
                y_ln = int(int(frc_msk[msk] * imgMsk.size(2) * imgMsk.size(2)) / (x_ln))

                x1 = torch.randint(low=0, high=int(imgMsk.size(2)-x_ln), size=(1,)).item()
                y1 = torch.randint(low=0, high=int(imgMsk.size(3)-y_ln), size=(1,)).item()

                if msk_typ == 0:
                    #imgMsk[:, :, x1:x1+x_ln, y1:y1+y_ln] = 0.
                    imgMsk[:, :, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))] = 0.

                elif msk_typ == 1:
                    '''imgMsk[:, 0, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
                    imgMsk[:, 1, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()
                    imgMsk[:, 2, x1:x1 + x_ln, y1:y1 + y_ln] = torch.rand(1).item()'''
                    imgMsk[:, 0, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))] = torch.rand(1).item()
                    imgMsk[:, 1, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))] = torch.rand(1).item()
                    imgMsk[:, 2, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))] = torch.rand(1).item()

                else:
                    #imgMsk[:, :, x1:x1+x_ln, y1:y1+y_ln] = torch.rand_like(imgMsk[:, :, x1:x1+x_ln, y1:y1+y_ln])
                    mask = torch.rand_like(imgMsk[:, :, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))])
                    imgMsk[:, :, 0:int(frc_msk[msk] * imgMsk.size(2)), 0:int(frc_msk[msk] * imgMsk.size(2))] = mask


                '''plt.imshow(imgMsk[0].to('cuda').permute(1,2,0))
                plt.show()'''

                #Perform recall free up gpu memory
                # Recall and free up gpu memory
                imgMsk = imgMsk.to('cuda')
                if infer_type == 0:
                    p = mem_unit.infer_max(imgMsk)[0].to('cpu')
                elif infer_type == 1:
                    p = mem_unit.infer_argmax(imgMsk)[0].to('cpu')
                elif infer_type == 2:
                    p = mem_unit.infer_softmax(imgMsk)[0].to('cpu')

                rcll_acc[s, msk] += ((torch.mean( torch.square((mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1))),
                                    dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s, msk] += torch.mean(torch.square(mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1)))

                p.to('cpu')
                imgMsk.to('cpu')
            mem_unit.to('cpu')

    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)


#Test recall
def recall(n_seeds, noise, frcmsk, rec_thr, test_t, hp_sz, mod_type=0, data=0, infer_type=0):
    with torch.no_grad():

        if test_t == 0:
            return noise_test(n_seeds, noise, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data, infer_type=infer_type)

        elif test_t == 1:
            return noise_test(n_seeds, noise, rec_thr, noise_tp=1, hip_sz=hp_sz[0], mod_type=mod_type, data=data, infer_type=infer_type)

        #Black Mask
        elif test_t == 2:
            return mask_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data, infer_type=infer_type)

        #Color Mask
        elif test_t == 3:
            return mask_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data, msk_typ=1, infer_type=infer_type)

        #Noise Mask
        elif test_t == 4:
            return mask_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data, msk_typ=2, infer_type=infer_type)



#Trains
def train(model_type=0, num_seeds=10, hip_sz=[1000], noise=[0], frcmsk=[0], test_t=0, rec_thr=.01, data=2, act=0, inf_type=0):
    with torch.no_grad():
        #Recall
        acc_means, acc_stds, mse_means, mse_stds = recall(num_seeds, noise=noise, frcmsk=frcmsk, rec_thr=rec_thr,
                                                          test_t=test_t, hp_sz=hip_sz, mod_type=model_type, data=data, infer_type=inf_type)

        print(f'Data:{data}', f'Test:{test_t}', 'InfType:', inf_type, f'\nAcc:{acc_means}', f'\nMSE:{mse_means}', '\n\n')


        with open(f'data/AutoAInferComp_infType{inf_type}_mtype{model_type}_ActF{act}_Test{test_t}_numN{hip_sz}_noise{noise}_frcMsk{frcmsk}_data{data}.data', 'wb') as filehandle:
            pickle.dump([acc_means, acc_stds, mse_means, mse_stds], filehandle)