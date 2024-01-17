import torch
from torch import nn
import math
import torch.nn.functional as F
import MHN
import random
from utilities import poisson as PoLU

mse = torch.nn.MSELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()
softmax = nn.Softmax(dim=1)

class LocalLayer(nn.Module):
    def __init__(self, in_dim=32, kernal_sz=8, in_chnls=3, out_chnls=1000, lr=.2, beta=10, optim=0):
        super().__init__()

        self.kernal_sz = kernal_sz                          # size of square kernal side, where stride=kernal size
        self.num_units = int((in_dim / kernal_sz)**2)       # num of units
        self.d = int(math.sqrt(self.num_units))
        self.in_channels = in_chnls
        self.out_channels = out_chnls
        self.lr = lr
        self.beta = beta
        self.units = self.create_units()



    def create_units(self):
        with torch.no_grad():
            units = torch.nn.ModuleList([torch.nn.ModuleList([]) for x in range(self.d)])
            for w in range(self.d):
                for h in range(self.d):
                    units[w].append(MHN.MemUnit(layer_szs=[self.kernal_sz**2 * self.in_channels, self.out_channels],
                                lr=self.lr, beta=self.beta, optim=3).to('cuda'))
        return units


    '''def softmax(self, input, beta=10):
        z = input.clone()
        for h in range(z.size(2)):
            for w in range(z.size(3)):
                idx = F.softmax(beta * z[:, :, w, h], dim=1)
                z[:, :, w, h] = idx
        return z'''


    def infer(self, input):
        a = torch.zeros(input.size(0), self.out_channels, self.d, self.d).to('cuda')
        for h in range(self.d):
            for w in range(self.d):
                ptch_inpt = input[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                            self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz].reshape(input.size(0), -1)

                a[:,:, w, h] = self.units[w][h].infer_step2(softmax(self.beta * ptch_inpt))
        return a


    def predict(self, z):
        p = torch.zeros(z.size(0), self.in_channels, z.size(2)*self.kernal_sz, z.size(2)*self.kernal_sz).to('cuda')
        for h in range(self.d):
            for w in range(self.d):
                pred = self.units[w][h].predict(softmax(self.beta * z[:,:, w, h]))
                p[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                            self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz] = pred.reshape(p.size(0), self.in_channels, self.kernal_sz, self.kernal_sz)

        return p

