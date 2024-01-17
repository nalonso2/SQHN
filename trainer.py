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
from autoassociate import train
import aa_online
import aa_onlineBP
import noisy_aa_online
import noisy_aa_onlineBP
import noisy_aa_onlineTree
import noisy_aa_onlineBPTree
import recognition
import recognition_MHN
import recognition_tree
import aa_online_tree
import aa_online_BPtree
import emerge_anylz
import heteroassociate



''' 
This file contains function for running the tests presented in the paper. These functions are used
by the main function to run each test with the hyperparameters used in the paper. No hyperparam
searches or adjustments are needed.
'''


#Initial Auto and Hetero Associative Memory Tests
def modCompareAA():
    # Comparison to PCN MHN
    for t in [0,1,2,3,4]:
        print(f'\n\n Moderate Corrupt   Model:{t}')
        for d in [2,4]:
            heteroassociate.train(model_type=t, test_t=2, hip_sz=[1024], noise=[.2], data=d, num_seeds=5)
            heteroassociate.train(model_type=t, test_t=0, hip_sz=[1024], frcmsk=[.25], data=d, num_seeds=5)
            heteroassociate.train(model_type=t, test_t=1, hip_sz=[1024], frcmsk=[.25], data=d, num_seeds=5)

    # Comparison to PCN MHN high corruption
    for t in [0,1,2,3,4]:
        print(f'\n\n High Corrupt   Model:{t}')
        for d in [2,4]:
            heteroassociate.train(model_type=t, test_t=2, hip_sz=[128], noise=[.8], data=d, num_seeds=7, rec_thr=.001)
            heteroassociate.train(model_type=t, test_t=0, hip_sz=[128], frcmsk=[.75], data=d, num_seeds=7, rec_thr=.001)
            heteroassociate.train(model_type=t, test_t=1, hip_sz=[128], frcmsk=[.75], data=d, num_seeds=7, rec_thr=.001)



#Online Continual Tests for One Hidden Layer
def online_L1():
    print('EMNIST')
    hidsz = [300, 1300, 2300]
    alpha = [5000, 25000, 45000]
    beta = [[.05, .025, .05, .05], [.1, .05, .1, .1], [.1, .05, .1, .1]]
    lr = [[.4, .01, .5, .9], [.35, .015, .45, .87], [.55, .015, .65, .85]]
    for hd in range(3):
        aa_online.train_online(784, hidsz[hd], simf=2, data=6, dev='cuda', max_iter=3000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7)
        aa_onlineBP.train_online(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7)
        aa_onlineBP.train_online(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7)
        aa_onlineBP.train_online(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], opt=2, num_seeds=7, r=1)
        aa_onlineBP.train_online(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7)

    print('\n\n\nCIFAR')
    hidsz = [700, 2000, 3300]
    alpha = [25000, 90000, 100000]
    beta = [[.005, .005, .005, .005], [.005, .005, .005, .005], [.01, .005, .01, .01]]
    lr = [[.5, .03, .6, .9], [.75, .03, .85, .85], [.75, .025, .85, .75]]
    for hd in range(3):
        aa_online.train_online(3072, hidsz[hd], simf=2, data=4, dev='cuda', max_iter=5000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7, t_fq=200)
        aa_onlineBP.train_online(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7, t_fq=200)
        aa_onlineBP.train_online(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7, t_fq=200)
        aa_onlineBP.train_online(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], opt=2, num_seeds=7, t_fq=200, r=1)
        aa_onlineBP.train_online(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7, t_fq=200)

    print('\n\n\nMNIST-FMNIST')
    hidsz = [300, 1300, 2300]
    alpha = [15000, 50000, 120000]
    beta = [[.01, .025, .01, .01], [.025, .025, .025, .025], [.025, .05, .025, .025]]
    lr = [[.4, .01, .4, .85], [.6, .015, .6, .85], [.65, .01, .7, .85]]
    for hd in range(3):
        aa_online.train_onContDom(784, hidsz[hd], simf=1, dev='cuda', max_iter=3000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7, cont=False)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7, cont=False)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7, cont=False)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], opt=2, r=.5, num_seeds=7, cont=False)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7, cont=False)





#Online Continual Tests for One Hidden layer
def onCont_L1():
    print('EMNIST')
    hidsz = [300, 1300, 2300]
    alpha = [5000, 22000, 50000]
    beta = [[.1, .1, .05, .1], [.1, .05, .05, .1], [.1, .05, .05, .1]]
    lr = [[.25, .005, .5, 1], [.35, .015, 1.3, .9], [.5, .015, 1.3, .85]]
    for hd in range(3):
        aa_online.train_onCont(784, hidsz[hd], simf=2, data=6, dev='cuda', max_iter=3000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7)
        aa_onlineBP.train_onCont(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7)
        aa_onlineBP.train_onCont(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7)
        aa_onlineBP.train_onCont(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], opt=2, num_seeds=7)
        aa_onlineBP.train_onCont(784, hidsz[hd], max_iter=3000, data=6, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7)

    # CIFAR-100
    print('\n\nCIFAR-100')
    hidsz = [700, 2000, 3300]
    alpha = [20000, 55000, 65000]
    beta = [[.01, .005, .01, .01], [.005, .005, .005, .005], [.005, .005, .005, .005]]
    lr = [[.5, .01, .6, .75], [.5, .025, .7, .85], [.5, .025, .7, .9]]
    for hd in range(3):
        aa_online.train_onCont(3072, hidsz[hd], simf=2, data=4, dev='cuda', max_iter=5000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7, t_fq=200)
        aa_onlineBP.train_onCont(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7, t_fq=200)
        aa_onlineBP.train_onCont(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7, t_fq=200)
        aa_onlineBP.train_onCont(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], opt=2, num_seeds=7, t_fq=200, r=.5)
        aa_onlineBP.train_onCont(3072, hidsz[hd], max_iter=5000, data=4, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7, t_fq=200)

    # MNIST-FMNIST
    print('\n\nMNIST-FMNIST')
    hidsz = [300, 1300, 2300]
    alpha = alpha = [8000, 50000, 90000]
    beta = [[.01, .05, .01,.01], [.01, .05, .01, .01], [.01, .05, .01, .01]]
    lr = [[.35, .01, .6, 1.2], [.5, .01, 1, 1.2], [.8, .01, 1.2, .95]]
    r = [5, 5, 5]
    for hd in range(3):
        aa_online.train_onContDom(784, hidsz[hd], simf=2, dev='cuda', max_iter=3000, wtupType=0, alpha=alpha[hd], det_type=0, num_seeds=7)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][0], lr=lr[hd][0], opt=0, num_seeds=7)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][1], lr=lr[hd][1], opt=1, num_seeds=7)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][2], lr=lr[hd][2], r=r[hd], opt=2, num_seeds=7)
        aa_onlineBP.train_onContDom(784, hidsz[hd], max_iter=3000, dev='cuda', beta=beta[hd][3], lr=lr[hd][3], opt=3, num_seeds=7)


#Run online-continual tests for one hidden layer models
def run_onCont_L1():
    #First get online data
    online_L1()
    #Then online continual
    onCont_L1()




#Ablation Sutdy (see supplementals)
def onlCont_Ablation():
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=0, alpha=12000, det_type=3, num_seeds=7)
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=0, alpha=14000, det_type=0, num_seeds=7)
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=1, alpha=14000, det_type=3, num_seeds=7)
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=0, alpha=.9, det_type=1, num_seeds=7)
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=2, alpha=2000, det_type=3, num_seeds=7)
    aa_online.train_onCont(3072, 700, simf=1, data=4, dev='cuda', max_iter=5000, wtupType=3, alpha=22000, det_type=3, num_seeds=7, lr=.5)




#Noisy Encoding Tests
def nsEncode():
    niter = [1, 5, 20, 50]
    l = [[.92, .92, .92, .94], [.9, .85, .85, .86]]
    for ns in range(2):
        for nit in range(4):
            print(f'NoiseType:{ns}  NumUpdates:{niter[nit]}')
            noisy_aa_online.train_online(784, 300, simf=2, data=6, dev='cuda', max_iter=300, wtupType=0, alpha=1000000,
                                    det_type=0, num_seeds=7, num_up=niter[nit], lr=l[ns][nit], ns_type=ns)

            noisy_aa_online.train_online(784, 300, simf=2, data=6, dev='cuda', max_iter=300, wtupType=0,
                                alpha=1000000, det_type=0, num_seeds=7, num_up=niter[nit], lr=1., ns_type=ns, plus=True)


    #BP-SGD
    b = [[.05, .05, .05, .05], [1, .05, .05, .05]]
    l = [[.4, .8, .6, .6], [.05, .8, .5, .5]]
    for ns in range(2):
        for nit in range(4):
            print(f'NoiseType:{ns}  NumUpdates:{niter[nit]}')
            noisy_aa_onlineBP.train_online(784, 300, max_iter=300, data=6, dev='cuda', beta=b[ns][nit], lr=l[ns][nit], opt=0,
                                         num_seeds=7, ns_type=ns, num_up=niter[nit])


    #BP-Adam
    b = [[.05, .05, .05, .05], [.05,.05,.05,.05]]
    l = [[.05, .01, .005, .001], [.05,.01,.005,.005]]
    for ns in range(2):
        for nit in range(4):
            print(f'NoiseType:{ns}  NumUpdates:{niter[nit]}')
            noisy_aa_onlineBP.train_online(784, 300, max_iter=300, data=6, dev='cuda', beta=b[ns][nit], lr=l[ns][nit], opt=1,
                                     num_seeds=7, ns_type=ns, num_up=niter[nit])




#Episodic Recognition Tests
def recog():
    # IPHN
    recognition.train_online(784, 300, simf=2, dev='cuda', max_iter=3000, num_seeds=7, alpha=50000, gamma=.99999, rec_type=0)
    recognition.train_online(784, 300, simf=2, dev='cuda', max_iter=3000, num_seeds=7, alpha=50000, gamma=.5, rec_type=1)

    # MHN
    recognition_MHN.train_online(784, 300, dev='cuda', max_iter=3000, num_seeds=7, gamma=70, beta=.05, lr=.01, opt=1, rec_type=0)
    recognition_MHN.train_online(784, 300, dev='cuda', max_iter=3000, num_seeds=7, gamma=.5, beta=.05, lr=.01, opt=1, rec_type=1)





#Online training for three layer (used for order sensitivity measure)
def online_L3():
    nd_sz = [200, 600, 1000]
    alphas = [8000, 22000, 50000]
    betas = [[1000, 1], [1000, 1], [8000, 1]]
    lrs = [[.9, .01, .9, 1], [.9, .01, .9, 1], [.8, .01, .8, .925]]
    for nd in range(3):
        print(f'\nCIFAR Online   Node:{nd_sz[nd]}')
        aa_online_tree.train_online(arch=2, data=4, max_iter=2000, wtupType=0, num_seeds=5, alpha=alphas[nd],
                            shuf=True, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd], run_test=True, save_md=False)

        aa_online_BPtree.train_online(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                            in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=0, lr=lrs[nd][0], run_test=True,
                                  save_md=False)

        aa_online_BPtree.train_online(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32, in_chn=3,
                                  chnls=nd_sz[nd], beta=betas[nd][1], optim=1, lr=lrs[nd][1], run_test=True, save_md=False)

        aa_online_BPtree.train_online(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                                      in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=2, lr=lrs[nd][2], run_test=True,
                                      save_md=False, r=.01)

        aa_online_BPtree.train_online(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                                      in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=3, lr=lrs[nd][3],
                                      run_test=True, save_md=False)



    # CIFAR-SVHN
    nd_sz = [200, 600, 1000]
    alphas = [16000, 35000, 88000]
    betas = [[10000, 10], [10000, 100], [10000, 1000]]
    lrs = [[.9, .01, .9, .9], [.7, .01, .7, .85], [.7, .012, .7, .8]]
    for nd in range(3):
        print(f'\nCIFAR-SVHN Online  Ndsz:{nd_sz[nd]}')
        aa_online_tree.train_onContDom(arch=2, max_iter=2000, wtupType=0, num_seeds=5, alpha=alphas[nd],
                        shuf=False, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd], cont=False)

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                              in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=0, lr=lrs[nd][0], cont=False)

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                              in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][1], optim=1, lr=lrs[nd][1], cont=False)

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                                     in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=2, lr=lrs[nd][2],
                                     cont=False, r=.01)

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                                         in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=3,
                                         lr=lrs[nd][3], cont=False)





#Online-Continual Training for Three layer
def onCont_L3():
    nd_sz = [200, 600, 1000]
    alphas = [10000, 30000, 62000]
    betas = [[1000, 50], [1000, 100], [8000, 100]]
    lrs = [[.75, .01, .8, .95], [.9, .005, .9, .95], [.9, .005, .9, .95]]
    r = [.05, .025, .015]
    for nd in range(3):
        print(f'\nCIFAR OCI   Node:{nd_sz[nd]}')
        aa_online_tree.train_onCont(arch=2, data=4, max_iter=2000, wtupType=0, num_seeds=5, alpha=alphas[nd],
                        shuf=True, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd], run_test=True, save_md=False)

        aa_online_BPtree.train_onCont(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                        in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=0, lr=lrs[nd][0], run_test=True, save_md=False)

        aa_online_BPtree.train_onCont(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                      in_chn=3, chnls=nd_sz[nd], beta=betas[nd][1], optim=1, lr=lrs[nd][1], run_test=True, save_md=False)

        aa_online_BPtree.train_onCont(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                                  in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=2, lr=lrs[nd][2], run_test=True,
                                  save_md=False, r=r[nd])

        aa_online_BPtree.train_onCont(arch=1, data=4, max_iter=2000, num_seeds=5, shuf=True, t_fq=100, in_dim=32,
                              in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=3, lr=lr[nd][3], run_test=True, save_md=False)

    # CIFAR-SVHN
    nd_sz = [200, 600, 1000]
    alphas = [15000, 45000, 88000]
    betas = [[10000, 100], [10000, 1000], [10000, 1000]]
    lrs = [[.7, .001, .85, .9], [.7, .001, .8, .85], [.9, .001, .75, .85]]
    r = [.1, .25, 1]
    for nd in range(3):
        print(f'\nCIFAR-SVHN ODI   Node:{nd_sz[nd]}')
        aa_online_tree.train_onContDom(arch=2, max_iter=2000, wtupType=0, num_seeds=5, alpha=alphas[nd],
                        shuf=True, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd])

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                              in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=0, lr=lrs[nd][0])

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                              in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][1], optim=1, lr=lrs[nd][1])

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                                     in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=2,
                                         lr=lrs[nd][2], r=r[nd])

        aa_online_BPtree.train_onContDom(arch=1, max_iter=2000, num_seeds=5, shuf=True, t_fq=100,
                                         in_dim=32, in_chn=3, chnls=nd_sz[nd], beta=betas[nd][0], optim=3, lr=lrs[nd][3])





#Get Online and Online-Continual data for the L3 SQHN model
def run_onCont_L3():
    #First get online data
    online_L3()
    #Then online-continual
    onCont_L3()




#Noisy encoding/learning with nopisy input tests (3 level model)
def nsEncode_L3():
    niter = [1, 10, 20]
    # SQHN
    l = [[.98, .85, .8], [.98, .6, .6]]
    for ns in range(2):
        for nit in range(3):
            noisy_aa_onlineTree.train_online(in_dim=32, arch=2, data=4, dev='cuda', max_iter=150, alpha=1000000, num_seeds=5,
                                         chnls=150, in_chn=3, num_up=niter[nit], lr=l[ns][nit], ns_type=ns)

            noisy_aa_onlineTree.train_online(in_dim=32, arch=2, data=4, dev='cuda', max_iter=150, alpha=1000000, num_seeds=5,
                                         chnls=150, in_chn=3, num_up=niter[nit], lr=1., ns_type=ns, plus=True)

    # BP-SGD
    b = [[10, 100, 100], [10, 100, 100]]
    l = [[.75, .8, .8], [.8,.8,.8]]
    for ns in range(2):
        for nit in range(3):
            noisy_aa_onlineBPTree.train_online(arch=1, data=4, max_iter=150, num_seeds=5, in_dim=32,
                        in_chn=3, chnls=150, beta=b[ns][nit], optim=0, lr=l[ns][nit], num_up=niter[nit], ns_type=ns)

    # BP-Adam
    b = [[.1, 1, 100], [100, 100, 100]]
    l = [[.01, .001, .001], [.005, .001, .001]]
    for ns in range(2):
        for nit in range(3):
            noisy_aa_onlineBPTree.train_online(arch=1, data=4, max_iter=150, num_seeds=5, in_dim=32,
                        in_chn=3, chnls=150, beta=b[ns][nit], optim=1, lr=l[ns][nit], num_up=niter[nit], ns_type=ns)






#### Compare architecture with 1, 2, or 3 hidden layers ####

#Compare auto and hetero-association
def arch_compare_AA():
    # Architecture Compare
    for d in [2,4,5]:
        for t in [4,7,8,9]:
            print('\n')
            for m in [2,3,4]:
                train(model_type=m, test_t=t, noise=[0, .05, .15, .25, .4, .5, .75, 1, 1.25, 1.5],
                      frcmsk=[0,.1, .25, .5, .75, 7/8, 15/16], data=d, num_seeds=10)


#Compare recognition
def arch_compare_recog():
    # SQHN L1
    recognition.train_online(3072, 500, simf=2, dev='cuda', max_iter=3000, num_seeds=7, alpha=500000, gamma=.99999, rec_type=0, data=1)
    recognition.train_online(3072, 500, simf=2, dev='cuda', max_iter=3000, num_seeds=7, alpha=500000, gamma=.5, rec_type=1, data=1)

    # SQHN L2 and L3
    recognition_tree.train_online(max_iter=3000, chnls=500, num_seeds=5, arch=5)
    recognition_tree.train_online(max_iter=3000, chnls=500, num_seeds=5, arch=6)


#Compare recognition w/ noise
def arch_compare_recog_noise():
    recognition.train_online(3072, 500, simf=2, dev='cuda', max_iter=3000, num_seeds=5, alpha=500000, gamma=.85, rec_type=0, data=1, noise=.2)
    recognition_tree.train_online(max_iter=3000, chnls=500, num_seeds=5, arch=5, noise=.2, gamma=.45)
    recognition_tree.train_online(max_iter=3000, chnls=500, num_seeds=5, arch=6, noise=.2, gamma=.25)


#Compare Online Learning
def arch_compare_online():
    nd_sz = [200, 600, 1000]
    mxit = [2500, 3000, 3500]
    for nd in range(3):
        #L1
        aa_online.train_online(3072, nd_sz[nd], simf=2, data=4, dev='cuda', max_iter=mxit[nd], wtupType=0,
                               alpha=50000, det_type=0, num_seeds=5, shuf=True)
        #L2
        aa_online_tree.train_online(arch=9, data=4, max_iter=mxit[nd], wtupType=0, num_seeds=5, alpha=50000,
                                shuf=True, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd])
        #L3
        aa_online_tree.train_online(arch=8, data=4, max_iter=mxit[nd], wtupType=0, num_seeds=5, alpha=100000,
                                shuf=True, t_fq=100, in_dim=32, in_chn=3, chnls=nd_sz[nd])


#Run all comparisons
def run_arch_compare():
    arch_compare_AA()
    arch_compare_online()
    arch_compare_recog()
    arch_compare_recog_noise()


#Run analysis of emergent properties during training (see supplementals)
def emerge():
    emerge_anylz.train_online(arch=8, data=5, max_iter=2000, wtupType=0, num_seeds=3, alpha=200,
                                shuf=True, t_fq=100, in_dim=64, in_chn=3, chnls=1000)