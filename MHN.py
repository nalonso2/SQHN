import torch
from torch import nn
import math
import torch.nn.functional as F
import random


softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()


class MemUnit(nn.Module):
    def __init__(self, layer_szs=[3072, 100], n_iter=1, lr=.01, binary=False, n=1, beta=1, gamma=.5, dev='cuda',
                 optim=0, lmbda=.5, mem_sz=64):
        super().__init__()

        self.layer_szs = layer_szs
        self.num_layers = len(layer_szs)
        self.n_iter = n_iter
        self.l_rate = lr
        self.wts = self.create_wts()
        self.binary = binary
        self.opt_type = optim
        self.n = n
        self.age = 0
        self.prior = torch.zeros(layer_szs[1]).to(dev)
        self.alpha = .7
        self.err_avg = 0
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lmbda
        self.mem = torch.zeros(0,layer_szs[0]).to(dev)
        self.mem_sz = mem_sz

        if optim == 0 or optim == 3:
            self.optim = torch.optim.SGD(self.wts.parameters(), lr=self.l_rate)
        elif optim == 1:
            self.optim = torch.optim.Adam(self.wts.parameters(), lr=self.l_rate)
        else:
            self.optim = None




    def create_wts(self):
        wts = nn.Linear(self.layer_szs[1], self.layer_szs[0], bias=False)
        return wts


    def load_wts(self, imgs, T=7000):
        for t in range(T):
            z, p = self.recall_step(imgs.detach())
            self.optim.zero_grad()
            loss = torch.mean(torch.square(imgs - p).sum(1))
            loss.backward()
            self.optim.step()
            #print(t, loss / imgs.size(1))


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

    def infer_step2(self, targ):
        with torch.no_grad():
            a = targ.matmul(self.wts.weight)
            return a


    def update_err_avg(self, targ):
        with torch.no_grad():
            self.age += 1
            z = self.infer_step(targ)
            t_hat = self.wts(z)
            err = torch.mean(torch.square(targ - t_hat))

            #self.prior += (z - self.prior) / (self.age)
            self.err_avg += (err - self.err_avg) / self.age



    def recognize(self, targ):
        with torch.no_grad():
            t = targ.clone().view(targ.size(0), -1)
            a = self.infer_step(t)
            mxlk = torch.max(a, dim=1)[0]
            mxC = torch.argmax(a, dim=1)
            return mxlk >= (self.prior[0,mxC] * self.gamma)



    def recognize2(self, targ):
        with torch.no_grad():
            t = targ.clone().view(targ.size(0), -1)
            z = self.infer_step(t)
            err = torch.mean(torch.square(t - self.wts(z)), dim=1)
            return err <= (self.err_avg * self.gamma)



    def recall_learn(self, inpt, targ=None):
        with torch.no_grad():
            if targ is None:
                t = inpt.clone().detach()
            else:
                t = targ.clone().detach()

        if self.opt_type < 3:
            z, p = self.recall_step(inpt.detach())
            self.optim.zero_grad()
            loss = torch.mean(torch.square(t - p).sum(1))
            loss.backward()
            self.optim.step()
        else:
            i = self.get_mem_batch(inpt)
            z, p = self.recall_step(i.detach())
            self.optim.zero_grad()
            loss = torch.mean(torch.square(i - p).sum(1))
            loss.backward()
            self.optim.step()
            self.update_mem(inpt)

        with torch.no_grad():
            self.age += 1
            self.prior = (1 / self.age) * torch.mean(z, dim=0) + (1 - 1 / self.age) * self.prior

        return z, t, loss




    def recall(self, targ):
        t = targ.clone().view(targ.size(0),-1)
        z, p = self.recall_step(t)
        return p




    #Resevoir memory updating for online (mini-batch size one) scenario
    def update_mem(self, inpt):
        with torch.no_grad():
            if self.mem.size(0) < self.mem_sz:
                self.mem = torch.cat((self.mem, inpt.clone()), dim=0)
            else:
                n = random.randint(0, self.age + 1)
                if n < self.mem_sz:
                    self.mem[n] = inpt[0]



    def get_mem_batch(self, inpt):
        return torch.cat((self.mem, inpt), dim=0)
