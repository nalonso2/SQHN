import torch
from torch import nn
import math
import torch.nn.functional as F
import random
import MHN_layer
from utilities import poisson as PoLU

softmax = nn.Softmax(dim=1)
mse = torch.nn.MSELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=1)
relu = nn.ReLU()


class Tree(nn.Module):
    def __init__(self, in_dim=64, chnls=200, n_iter=2, lr=1, arch=0, in_chnls=3, lmbd=.5, beta=10, r=.1):
        super().__init__()

        self.in_dim = in_dim
        self.arch = arch
        self.channels = chnls
        self.in_chnls = in_chnls
        self.n_iter = n_iter
        self.r = r
        self.lr = lr
        self.beta = beta
        self.layers = self.create_layers()
        self.lmbd = lmbd
        self.age = 0
        self.optim = torch.optim.SGD(self.layers.parameters(), lr=self.lr)

        self.mean = {}
        self.F = {}
        for n, p in self.layers.named_parameters():
            self.mean[n] = p.data.clone()
            self.F[n] = torch.zeros_like(p.data)


    def create_layers(self):
        with torch.no_grad():
            #Tiny ImageNet
            if self.arch == 0:
                layers = torch.nn.ModuleList([MHN_layer.LocalLayer(in_dim=self.in_dim, kernal_sz=8, in_chnls=self.in_chnls,
                                                                   out_chnls=self.channels, beta=self.beta),

                          MHN_layer.LocalLayer(in_dim=int(self.in_dim / 8), kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, beta=self.beta),

                          MHN_layer.LocalLayer(in_dim=int(self.in_dim / 32), kernal_sz=int(self.in_dim / 32), in_chnls=self.channels,
                                              out_chnls=self.channels, beta=self.beta)])


            #CIFAR
            elif self.arch == 1:
                layers = torch.nn.ModuleList([MHN_layer.LocalLayer(in_dim=self.in_dim, kernal_sz=4, in_chnls=3,
                                                                   out_chnls=self.channels, beta=self.beta),

                          MHN_layer.LocalLayer(in_dim=int(self.in_dim / 4), kernal_sz=4, in_chnls=self.channels,
                                              out_chnls=self.channels, beta=self.beta),

                          MHN_layer.LocalLayer(in_dim=int(self.in_dim / 16), kernal_sz=int(self.in_dim / 16),
                                              in_chnls=self.channels, out_chnls=self.channels, beta=self.beta)])

            return layers



    def ff(self, input):
        i = [0 for x in range(len(self.layers) + 1)]
        i[0] = input.clone()
        i[1] = self.layers[0].infer(i[0])

        for l in range(1, len(self.layers)):
            i[l+1] = self.layers[l].infer(i[l])

        return i



    def predict(self, a):
        z = a[-1]
        for l in reversed(range(1,len(self.layers))):
            z = self.lmbd * a[l] + (1 - self.lmbd) * self.layers[l].predict(z)
        z = self.layers[0].predict(z)
        return z


    def recall(self, input):
        x = input.clone()
        a = self.ff(x)
        out = self.predict(a)
        return out


    def recall_learn(self, input):
        p = self.recall(input)
        self.optim.zero_grad()
        loss = torch.mean(torch.square(input.detach() - p).sum(1))
        for n, p in self.layers.named_parameters():
            loss += (self.r * self.F[n].detach() * (p - self.mean[n].detach())**2).sum()
        loss.backward()
        self.optim.step()

        with torch.no_grad():
            self.age += 1
        self.update_mean()
        self.update_fisher(input)


    def update_fisher(self, x):
        self.optim.zero_grad()
        p = self.recall(x.detach())
        loss = torch.mean(torch.square(x.detach() - p).sum(1))
        loss.backward()

        with torch.no_grad():
            v = max(1 / (self.age), .001)
            for n, p in self.layers.named_parameters():
                self.F[n] = v * p.grad.data.clone()**2 + (1 - v) * self.F[n]

        self.optim.zero_grad()


    def update_mean(self):
        with torch.no_grad():
            v = max(1 / (self.age+1), .001)
            for n, p in self.layers.named_parameters():
                self.mean[n] = v * p.data.clone() + (1 - v) * self.mean[n]