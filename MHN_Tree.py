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
    def __init__(self, in_dim=64, chnls=200, n_iter=2, lr=1, arch=0, in_chnls=3, lmbd=.5, optim=0, beta=10, mem_sz=64, dev='cuda'):
        super().__init__()

        self.in_dim = in_dim
        self.arch = arch
        self.channels = chnls
        self.in_chnls = in_chnls
        self.n_iter = n_iter
        self.lr = lr
        self.beta = beta
        self.layers = self.create_layers()
        self.lmbd = lmbd
        self.mem = torch.zeros(0, 3, in_dim, in_dim).to(dev)
        self.mem_sz = mem_sz
        self.age = 0
        self.opt_type = optim

        if optim == 0 or optim == 3:
            self.optim = torch.optim.SGD(self.layers.parameters(), lr=self.lr)
        elif optim == 1:
            self.optim = torch.optim.Adam(self.layers.parameters(), lr=self.lr)



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


    def recall_learn(self, input, target=None):
        if target == None:
            t = input.detach()
        else:
            t = target.detach()

        if self.opt_type < 3:
            z, p = self.recall(inpt.detach())
            self.optim.zero_grad()
            loss = torch.mean(torch.square(t - p).sum(1))
            loss.backward()
            self.optim.step()
        else:
            i = self.get_mem_batch(input)
            p = self.recall(i.detach())
            self.optim.zero_grad()
            loss = torch.mean(torch.square(i - p).sum(1))
            loss.backward()
            self.optim.step()
            self.update_mem(input)

        self.age += 1



    # Resevoir memory updating for online (mini-batch size one) scenario
    def update_mem(self, inpt):
            with torch.no_grad():
                if self.mem.size(0) < self.mem_sz:
                    self.mem = torch.cat((self.mem, inpt.clone()), dim=0)
                else:
                    n = random.randint(0, self.age + 1)
                    if n < self.mem_sz:
                        self.mem[n] = inpt[0]

    #Concatenate input with memory
    def get_mem_batch(self, inpt):
        return torch.cat((self.mem, inpt), dim=0)


