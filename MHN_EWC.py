import torch
from torch import nn
import math
import torch.nn.functional as F
from copy import deepcopy

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


class MemUnit(nn.Module):
    def __init__(self, layer_szs=[3072, 100], n_iter=1, lr=.01, binary=False, n=1, beta=1, gamma=.5, r=.5, dev='cuda'):
        super().__init__()

        self.layer_szs = layer_szs
        self.num_layers = len(layer_szs)
        self.n_iter = n_iter
        self.l_rate = lr
        self.wts = self.create_wts()
        self.binary = binary
        self.n = n
        self.age = 0
        self.prior = torch.zeros(layer_szs[1]).to(dev)
        self.err_avg = 0
        self.beta = beta
        self.gamma = gamma
        self.r = r
        self.optim = torch.optim.SGD(self.wts.parameters(), lr=self.l_rate)
        self.mean = self.wts.weight.data.clone().to(dev)
        self.prcn_mtx = torch.zeros_like(self.wts.weight.data).to(dev)


    def create_wts(self):
        wts = nn.Linear(self.layer_szs[1], self.layer_szs[0], bias=False)
        return wts


    def update_fisher(self, x):
        self.optim.zero_grad()
        _, p = self.recall_step(x.detach())
        loss = torch.mean(torch.square(x.detach() - p).sum(1))
        loss.backward()
        with torch.no_grad():
            v = max(1 / (self.age), .001)
            self.prcn_mtx = v * self.wts.weight.grad.data**2 + (1 - v) * self.prcn_mtx

        self.optim.zero_grad()


    def update_mean(self):
        with torch.no_grad():
            v = max(1 / (self.age+1), .001)
            self.mean = v * self.wts.weight.data.clone() + (1 - v) * self.mean



    def recall_step(self, targ):
        a = targ.matmul(self.wts.weight)
        z = softmax(self.beta * a)
        return z, self.wts(z)


    def predict(self, z):
        return self.wts(z)


    def infer_step(self, targ):
        with torch.no_grad():
            a = targ.matmul(self.wts.weight)
            z = softmax(self.beta * a)
            return z



    def recall_learn(self, inpt, targ=None):
        i = inpt.clone().detach()
        with torch.no_grad():
            if targ is None:
                t = inpt.clone().detach()
            else:
                t = targ.clone().detach()

        z, p = self.recall_step(i)

        self.optim.zero_grad()
        loss = torch.mean(torch.square(t - p).sum(1))
        loss += (self.r * self.prcn_mtx.detach() * (self.wts.weight - self.mean.detach())**2).sum()
        loss.backward()
        self.optim.step()

        with torch.no_grad():
            self.age += 1
        self.update_mean()
        self.update_fisher(i)

        return z, t, loss


    def recall(self, targ):
        t = targ.clone()
        z, p = self.recall_step(t)
        return p
