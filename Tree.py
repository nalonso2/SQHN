import torch
from torch import nn
import math
import torch.nn.functional as F
import random
import LocLayer


softmax = nn.Softmax(dim=1)
mse = torch.nn.MSELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=1)
relu = nn.ReLU()

class Tree(nn.Module):
    def __init__(self, in_dim=64, chnls=200, n_iter=2, lr=1, actFunc=0, simFunc=1, alpha=.8, wt_up=0,
                 arch=0, in_chnls=3, lmbd=.5, gamma=.99, b_sim=2):
        super().__init__()

        self.in_dim = in_dim
        self.arch = arch
        self.channels = chnls
        self.in_chnls = in_chnls
        self.n_iter = n_iter
        self.lr = lr
        self.wtUp = wt_up                  # 0=online avg, 1=online weighted avg, 2=online overwrite, 3=standard SGD
        self.simFunc = simFunc             # 0=dot, 1=cos, 2=mean_shift cos, 3=hetero_cos
        self.b_sim = b_sim
        self.actFunc = actFunc             # 0=max, 1=softmax, 2=mean-shifted
        self.alpha = alpha
        self.layers = self.create_layers()
        self.age = 0
        self.lmbd = lmbd
        self.gamma = gamma



    def create_layers(self):
        with torch.no_grad():
            if self.arch == 0:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=8, in_chnls=self.in_chnls, out_chnls=self.channels,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=int(self.in_dim / 8), kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=int(self.in_dim / 32), kernal_sz=int(self.in_dim / 32), in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]


            elif self.arch == 1:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=8, in_chnls=self.in_chnls, out_chnls=self.channels,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=int(self.in_dim / 8), kernal_sz=int(self.in_dim / 8), in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]




            elif self.arch == 2:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=4, in_chnls=3, out_chnls=self.channels,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=int(self.in_dim / 4), kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=int(self.in_dim / 16), kernal_sz=int(self.in_dim / 16),
                                              in_chnls=self.channels, out_chnls=self.channels, lr=self.lr,
                                              actFunc=self.actFunc, simFunc=self.simFunc, alpha=self.alpha,
                                              wt_up=self.wtUp)]



            #MNIST 2-layer
            elif self.arch == 3:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=4, in_chnls=1, out_chnls=20,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=7, kernal_sz=7, in_chnls=20, out_chnls=500, lr=self.lr,
                                              actFunc=self.actFunc, simFunc=self.simFunc,
                                              alpha=self.alpha, wt_up=self.wtUp)]


            #MNIST 3-layer
            elif self.arch == 4:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=2, in_chnls=1, out_chnls=6,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=14, kernal_sz=2, in_chnls=6, out_chnls=20, lr=self.lr,
                                              actFunc=self.actFunc, simFunc=self.simFunc,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=7, kernal_sz=7, in_chnls=20, out_chnls=500, lr=self.lr,
                                              actFunc=self.actFunc, simFunc=self.simFunc,
                                              alpha=self.alpha, wt_up=self.wtUp)
                          ]


            #CIFAR variable 2-layer
            elif self.arch == 5:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=4, in_chnls=self.in_chnls, out_chnls=40,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=8, kernal_sz=8, in_chnls=40,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]


            #CIFAR variable 3-layer
            elif self.arch == 6:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=2, in_chnls=self.in_chnls, out_chnls=40,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=16, kernal_sz=4, in_chnls=40,
                                              out_chnls=200, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=4, kernal_sz=4, in_chnls=200,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)
                          ]


            elif self.arch == 7:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=int(self.in_dim/4), in_chnls=self.in_chnls,
                                              out_chnls=self.channels,
                                              lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=4, kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]



            elif self.arch == 8:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=int(self.in_dim/16), in_chnls=self.in_chnls,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                              alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=16, kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp),

                          LocLayer.LocalLayer(in_dim=4, kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                              simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]


            elif self.arch == 9:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=int(self.in_dim / 8), in_chnls=self.in_chnls,
                                        out_chnls=self.channels,
                                        lr=self.lr, actFunc=self.actFunc, simFunc=self.b_sim,
                                        alpha=self.alpha, wt_up=self.wtUp),

                    LocLayer.LocalLayer(in_dim=8, kernal_sz=8, in_chnls=self.channels,
                                        out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                        simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]




            elif self.arch == 10:
                layers = [LocLayer.LocalLayer(in_dim=self.in_dim, kernal_sz=int(self.in_dim / 8), in_chnls=self.in_chnls,
                                        out_chnls=self.channels,
                                        lr=self.lr, actFunc=self.actFunc, simFunc=4,
                                        alpha=self.alpha, wt_up=self.wtUp),

                    LocLayer.LocalLayer(in_dim=8, kernal_sz=8, in_chnls=self.channels,
                                        out_chnls=self.channels, lr=self.lr, actFunc=self.actFunc,
                                        simFunc=self.simFunc, alpha=self.alpha, wt_up=self.wtUp)]


            return layers




    def load_wts(self, inbatch):
        i = inbatch
        self.layers[0].load_wts(i)
        i = self.arg_max(self.layers[0].infer(i))
        for l in range(1, len(self.layers)):
            self.layers[l].load_wts(i)
            i = self.arg_max(self.layers[l].infer(i))


    def max(self, input):
        with torch.no_grad():
            z = input.clone()
            for h in range(z.size(2)):
                for w in range(z.size(3)):
                    idx = F.one_hot(torch.argmax(z[:, :, w, h], dim=1), num_classes=z.size(1)).float().to('cuda')
                    z[:, :, w, h] *= idx
            return z


    def arg_max(self, input):
        with torch.no_grad():
            z = input.clone()
            for h in range(z.size(2)):
                for w in range(z.size(3)):
                    idx = F.one_hot(torch.argmax(z[:, :, w, h], dim=1), num_classes=z.size(1)).float().to('cuda')
                    z[:, :, w, h] = idx
            return z



    def ff_max(self, input):
        with torch.no_grad():
            i = [0 for x in range(len(self.layers) + 1)]
            i[0] = input.clone()
            i[1] = self.layers[0].infer(i[0])

            for l in range(1, len(self.layers)):
                i[l+1] = self.layers[l].infer_max(i[l])

            return i


    def ff_argmax(self, input):
        with torch.no_grad():
            i = [0 for x in range(len(self.layers) + 1)]
            i[0] = input.clone()
            i[1] = self.layers[0].infer(i[0])
            for l in range(1, len(self.layers)):
                i[l+1] = self.layers[l].infer_argmax(i[l])
            return i


    def ff_softmax(self, input, beta=10000):
        with torch.no_grad():
            i = [0 for x in range(len(self.layers) + 1)]
            i[0] = input.clone()
            i[1] = self.layers[0].infer(i[0])
            for l in range(1, len(self.layers)):
                i[l+1] = self.layers[l].infer_softmax(i[l], beta=beta)
            return i



    def predict_argmax(self, a):
        with torch.no_grad():
            z = a[-1]
            for l in reversed(range(1,len(self.layers))):
                z = self.lmbd * a[l] + (1 - self.lmbd) * self.layers[l].predict_argmax(z)
            z = self.layers[0].predict_argmax(z)
            return z



    def predict_max(self, a):
        with torch.no_grad():
            z = [0 for x in range(len(self.layers) + 1)]
            z[-1] = a[-1]
            for l in reversed(range(2, len(self.layers)+1)):
                z[l - 1] = self.lmbd * a[l] + (1 - self.lmbd) * self.layers[l].predict_argmax(z)
            z[0] = self.layers[0].predict_argmax(z[1])
            return z




    def infer_max(self, input):
        with torch.no_grad():
            a = self.ff_max(input)
            for l in reversed(range(2, len(self.layers)+1)):
                a[l - 1] = self.lmbd * a[l-1] + (1 - self.lmbd) * self.layers[l-1].predict_argmax(a[l])
            a[0] = self.layers[0].predict_argmax(a[1])
            return a


    def soft_infer_max(self, input):
        with torch.no_grad():
            a = self.ff_max(input)
            for l in reversed(range(2, len(self.layers)+1)):
                a[l - 1] = self.lmbd * a[l-1] + (1 - self.lmbd) * self.layers[l-1].predict_max(a[l])
            a[0] = self.layers[0].predict_argmax(a[1])
            return a



    def infer_max_iter(self, input, iter=5):
        with torch.no_grad():
            a = self.ff_max(input)
            for i in range(iter):
                #Predict
                for l in reversed(range(2, len(self.layers)+1)):
                    a[l - 1] = (self.lmbd) * a[l-1] + (1 - self.lmbd) * self.layers[l-1].predict_argmax(a[l])

                #Bottom Up
                for l in range(len(self.layers)):
                    a[l + 1] = (self.lmbd) * a[l+1] + (1 - self.lmbd) * self.layers[l].infer_max(a[l])

            a[0] = self.layers[0].predict_argmax(a[1])
            return a



    def infer_argmax(self, input):
        with torch.no_grad():
            a = self.ff_argmax(input)
            for l in reversed(range(2, len(self.layers)+1)):
                a[l - 1] = self.lmbd * a[l-1] + (1 - self.lmbd) * self.layers[l-1].predict_argmax(a[l])
            a[0] = self.layers[0].predict_argmax(a[1])
            return a



    def infer_softmax(self, input, beta=10000):
        with torch.no_grad():
            a = self.ff_softmax(input, beta=beta)
            for l in reversed(range(2, len(self.layers)+1)):
                a[l - 1] = self.lmbd * a[l-1] + (1 - self.lmbd) * self.layers[l-1].predict_argmax(a[l])
            a[0] = self.layers[0].predict_argmax(a[1])
            return a



    def recall(self, input, iter=1):
        with torch.no_grad():
            x = input.clone()
            for i in range(iter):
                a = self.ff_max(x)
                x = self.predict_argmax(a)
        return x



    def soft_recall(self, input):
        with torch.no_grad():
            x = input.clone()
            for i in range(iter):
                a = self.ff_max(x)
                x = self.predict_max(a)
        return x


    def update_wts(self, input):
        self.age += 1
        for l in range(len(self.layers)):
            a = self.ff_max(input)
            if l > 0:
                a[l] = self.arg_max(a[l])
            self.layers[l].update_wts(self.max(a[l+1]), a[l])


    def update_wts2(self, a):
        self.age += 1
        for l in range(len(self.layers)):
            if l > 0:
                a[l] = self.arg_max(a[l])
            self.layers[l].update_wts(self.max(a[l+1]), a[l])


    def update_layer(self, input, l):
        a = self.ff_max(input)
        if l > 0:
            a[l] = self.arg_max(a[l])
        self.layers[l].update_wts(self.max(a[l+1]), a[l])


    def recognize(self, input):
        a = self.ff_argmax(input)
        test = a[-1].clone()
        #a[-1] = self.layers[-1].infer_argmax(a[-2])
        a[-1] = a[-1].view(a[-1].size(0), -1)
        mxlk = torch.max(a[-1], dim=1)[0]
        mxC = torch.argmax(a[-1], dim=1)
        '''print(mxlk)'''
        return mxlk >= (self.layers[-1].units[0][0].prior[mxC] * self.gamma)


    def compute_energy(self, x, a):
        with torch.no_grad():
            E = 0
            #Num nodes = num hidden nodes - top node + num visible nodes
            num_nodes = sum([self.layers[x].num_units for x in range(len(self.layers))]) - 1 + self.layers[0].num_units

            #Get energy at hidden layers
            for l in reversed(range(1, len(self.layers))):
                E += ((self.arg_max(a[l]) * self.layers[l].predict_argmax(a[l+1])).sum() / a[l].size(0)).item()

            #Get energy at Output Layer
            ks = int(self.layers[0].kernal_sz)
            n_ks = int(x.size(2) / self.layers[0].kernal_sz)
            for h in range(n_ks):
                for w in range(n_ks):
                    im_ptch = x[:, :, h*ks:h*ks+ks, w*ks:w*ks+ks]
                    pred_ptch = self.layers[0].predict_argmax(a[1])[:, :, h*ks:h*ks+ks, w*ks:w*ks+ks]
                    E += torch.mean(cos((im_ptch-.5).reshape(im_ptch.size(0),-1), (pred_ptch-.5).reshape(im_ptch.size(0), -1)) * .5 + .5)
                    #E += torch.mean(im_ptch * pred_ptch + (1 - im_ptch) * (1 - pred_ptch))

            return E / num_nodes


    def get_nonZero_prms(self):
        with torch.no_grad():
            p = 0
            for l in range(len(self.layers)):
                p += self.layers[l].get_nonZero_params()
            return p


    def get_avg_num_nrns(self):
        numn = []
        for l in range(len(self.layers)):
            numn.append(self.layers[l].get_avg_num_nrns())

        return numn


    def get_avg_lr(self):
        lrs = []
        for l in range(len(self.layers)):
            lrs.append(self.layers[l].get_avg_lr())

        return lrs


    def get_num_nonzero_wts(self):
        nwts = []
        for l in range(len(self.layers)):
            nwts.append(self.layers[l].get_num_nonzero_wts())

        return nwts

