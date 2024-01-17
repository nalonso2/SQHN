import torch
from torch import nn
import math
import torch.nn.functional as F
import Unit
import random
from utilities import poisson as PoLU


mse = torch.nn.MSELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()

class LocalLayer(nn.Module):
    def __init__(self, in_dim=32, kernal_sz=8, in_chnls=3, out_chnls=1000, lr=1, actFunc=0, simFunc=0, beta=10,
                 alpha=.8, wt_up=0):
        super().__init__()

        self.kernal_sz = kernal_sz                          # size of square kernal side, where stride=kernal size
        self.num_units = int((in_dim / kernal_sz)**2)       # num of units
        self.d = int(math.sqrt(self.num_units))
        self.in_channels = in_chnls
        self.out_channels = out_chnls
        self.lr = lr
        self.wtUp = wt_up                                   # 0=online avg, 1=online weighted avg, 2=online overwrite, 3=standard SGD
        self.simFunc = simFunc                              # 0=dot, 1=cos, 2=meanshift-cos
        self.actFunc = actFunc                              # 0=max, 1=softmax
        self.alpha = alpha
        self.beta = beta
        self.units = self.create_units()



    def create_units(self):
        with torch.no_grad():
            units = [[] for x in range(self.d)]
            for w in range(self.d):
                for h in range(self.d):
                    units[w].append(Unit.MemUnit(layer_szs=[self.kernal_sz**2 * self.in_channels, self.out_channels],
                                lr=self.lr, actFunc=self.actFunc,  simFunc=self.simFunc, beta=self.beta,
                               alpha=self.alpha, wt_up=self.wtUp).to('cuda'))
        return units



    def load_wts(self, inbatch):
        with torch.no_grad():
            # (batch, channel, row, column)
            #z = torch.zeros(inbatch.size(0), self.out_channels, self.d, self.d).to('cuda')

            #Get each non-overlapping patch and store in column.
            for w in range(self.d):
                for h in range(self.d):
                    patch = inbatch[:, :, self.kernal_sz*w:self.kernal_sz*w + self.kernal_sz, self.kernal_sz*h:self.kernal_sz*h + self.kernal_sz]
                    self.units[w][h].load_wts(patch.reshape(inbatch.size(0), -1))




    def update_wts(self, lk, target):
        with torch.no_grad():
            for w in range(self.d):
                for h in range(self.d):
                    ptch_lk = lk[:, :, w, h].view(lk.size(0), -1)
                    ptch_z = F.one_hot(torch.argmax(ptch_lk, dim=1), num_classes=ptch_lk.size(1)).float().to('cuda')
                    ptch_targ = target[:, :, self.kernal_sz*w:self.kernal_sz*w + self.kernal_sz, self.kernal_sz*h:self.kernal_sz*h + self.kernal_sz].reshape(target.size(0), -1)
                    self.units[w][h].update_wts(ptch_lk, ptch_z, ptch_targ)





    def max(self, input):
        with torch.no_grad():
            z = input.clone()
            for h in range(z.size(2)):
                for w in range(z.size(3)):
                    idx = F.one_hot(torch.argmax(z[:, :, w, h], dim=1), num_classes=z.size(1)).float().to('cuda')
                    z[:, :, w, h] *= idx
            return z




    def argmax(self, input):
        with torch.no_grad():
            z = input.clone()
            for h in range(z.size(2)):
                for w in range(z.size(3)):
                    idx = F.one_hot(torch.argmax(z[:, :, w, h], dim=1), num_classes=z.size(1)).float().to('cuda')
                    z[:, :, w, h] = idx
            return z


    def softmax(self, input, beta=10):
        with torch.no_grad():
            z = input.clone()
            for h in range(z.size(2)):
                for w in range(z.size(3)):
                    idx = F.softmax(beta * z[:, :, w, h], dim=1)
                    z[:, :, w, h] = idx
            return z




    def infer(self, input):
        with torch.no_grad():
            a = torch.zeros(input.size(0), self.out_channels, self.d, self.d).to('cuda')

            for h in range(self.d):
                for w in range(self.d):
                    ptch_inpt = input[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz].reshape(input.size(0), -1)

                    a[:,:, w, h] = self.units[w][h].infer_step(ptch_inpt)
            return a





    def infer_max(self, input):
        with torch.no_grad():
            a = torch.zeros(input.size(0), self.out_channels, self.d, self.d).to('cuda')

            mx_in = self.max(input)
            for h in range(self.d):
                for w in range(self.d):
                    ptch_inpt = mx_in[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz].reshape(input.size(0), -1)
                    a[:,:, w, h] = self.units[w][h].infer_step(ptch_inpt)
            return a



    def infer_argmax(self, input):
        with torch.no_grad():
            a = torch.zeros(input.size(0), self.out_channels, self.d, self.d).to('cuda')

            argmx_in = self.argmax(input)
            for h in range(self.d):
                for w in range(self.d):
                    ptch_inpt = argmx_in[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz].reshape(input.size(0), -1)
                    a[:,:, w, h] = self.units[w][h].infer_step(ptch_inpt)
            return a


    def infer_softmax(self, input, beta=10):
        with torch.no_grad():
            a = torch.zeros(input.size(0), self.out_channels, self.d, self.d).to('cuda')

            sftmx_in = self.softmax(input, beta=beta)
            for h in range(self.d):
                for w in range(self.d):
                    ptch_inpt = sftmx_in[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz].reshape(input.size(0), -1)
                    a[:,:, w, h] = self.units[w][h].infer_step(ptch_inpt)
            return a


    def predict(self, z):
        with torch.no_grad():
            p = torch.zeros(z.size(0), self.in_channels, z.size(2)*self.kernal_sz, z.size(2)*self.kernal_sz).to('cuda')

            for h in range(self.d):
                for w in range(self.d):
                    pred = self.units[w][h].predict(z[:, :, w, h].reshape(z.size(0), -1))
                    p[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz] = pred.reshape(p.size(0), self.in_channels, self.kernal_sz, self.kernal_sz)

            return p


    def predict_argmax(self, z):
        with torch.no_grad():
            p = torch.zeros(z.size(0), self.in_channels, z.size(2)*self.kernal_sz, z.size(2)*self.kernal_sz).to('cuda')
            for h in range(self.d):
                for w in range(self.d):
                    agmx =  F.one_hot(torch.argmax(z[:,:, w, h], dim=1), num_classes=self.out_channels).float().to('cuda')
                    pred = self.units[w][h].predict(agmx)
                    p[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz] = pred.reshape(p.size(0), self.in_channels, self.kernal_sz, self.kernal_sz)

            return p


    def predict_max(self, z):
        with torch.no_grad():
            p = torch.zeros(z.size(0), self.in_channels, z.size(2)*self.kernal_sz, z.size(2)*self.kernal_sz).to('cuda')

            for h in range(self.d):
                for w in range(self.d):
                    agmx =  F.one_hot(torch.argmax(z[:,:, w, h], dim=1), num_classes=self.out_channels).float().to('cuda')
                    pred = self.units[w][h].predict(agmx * z[:,:, w, h])
                    p[:, :, self.kernal_sz * w:self.kernal_sz * w + self.kernal_sz,
                                self.kernal_sz * h:self.kernal_sz * h + self.kernal_sz] = pred.reshape(p.size(0), self.in_channels, self.kernal_sz, self.kernal_sz)

            return p


    def get_nonZero_params(self):
        p = 0
        for h in range(self.d):
            for w in range(self.d):
                p += (self.units[w][h].wts.weight.data > 0).sum()
        return p


    def get_avg_num_nrns(self):
        avg = 0
        n = 0
        for h in range(self.d):
            for w in range(self.d):
                avg += (self.units[w][h].counts > 0).sum().item()
                n += 1

        return avg / n


    def get_avg_lr(self):
        avg = 0
        n = 0
        for h in range(self.d):
            for w in range(self.d):
                g = (self.units[w][h].counts > 0).float()
                lr = (1 / (self.units[w][h].counts + .00001)) * g
                avg += lr.sum().item() / g.sum().item()
                n += 1

        return avg / n


    def get_avg_cnt(self):
        avg = 0
        n = 0
        for h in range(self.d):
            for w in range(self.d):
                temp = self.units[w][h].counts > 0
                idx = temp.nonzero()
                avg += (self.units[w][h].counts[idx]).sum().item() / self.units[w][h].counts[idx].size(0).item()
                n += 1

        return avg / n


    def get_num_nonzero_wts(self):
        avg = 0
        n = 0
        for h in range(self.d):
            for w in range(self.d):
                g = (self.units[w][h].counts > 0).sum().item()
                avg += (self.units[w][h].wts.weight.data > 0).sum().item() / (self.units[w][h].wts.weight.data.size(0) * g)
                n += 1

        return avg / n











############# Unit TESTS ###############
def test(num_imgs=1000, im_sz=64, corrupt_type=0):
    with torch.no_grad():

        if corrupt_type == 0:
            tp = 'Non-Corrupt'
        elif corrupt_type == 1:
            tp = 'White Noise'
        elif corrupt_type == 2:
            tp = 'Mask'



        layer1 = LocalLayer(in_dim=im_sz, kernal_sz=4, in_chnls=3, out_chnls=num_imgs, lr=.2, actFunc=0, simFunc=1,
                            beta=10, alpha=.8, wt_up=0)

        layer2 = LocalLayer(in_dim=16, kernal_sz=4, in_chnls=num_imgs, out_chnls=num_imgs, lr=.2, actFunc=0, simFunc=1,
                            beta=10, alpha=.8, wt_up=0)

        layer3 = LocalLayer(in_dim=4, kernal_sz=4, in_chnls=num_imgs, out_chnls=num_imgs, lr=.2, actFunc=0, simFunc=1,
                            beta=10, alpha=.8, wt_up=0)


        #Load Weights
        x = torch.rand(num_imgs, 3, im_sz, im_sz).to('cuda')
        layer1.load_wts(x)
        i1 = layer1.infer_max(x)
        layer2.load_wts(i1)
        i2 = layer2.infer_max(i1)
        layer3.load_wts(i2)


        #Recall
        if corrupt_type == 1:
            x_hat = torch.clamp(x + torch.randn_like(x) * 1.5, min=0, max=1)
        elif corrupt_type == 2:
            x_hat = x.clone()
            x_hat[:, :, 0:62, :] = 0
        else:
            x_hat = x.clone()


        i1 = layer1.infer_max(x_hat)
        #print(i1.shape)
        i2 = layer2.infer_max(i1)
        #print(i2.shape)
        i3 = layer3.infer_max(i2)
        #print(i3.shape)
        #print(torch.max(i3), torch.min(i3))

        p = layer3.predict_argmax(i3)
        #print(p.shape)
        p = layer2.predict_argmax(p)
        #print(p.shape)
        p = layer1.predict_argmax(p)
        #print(p.shape)

        print(f'{tp}, Recalled: '
              f'{(torch.mean(torch.square((x.reshape(x.size(0), -1) - p.reshape(x.size(0), -1))), dim=1) <= .001).sum()} / {num_imgs}')


'''test()
test(corrupt_type=1)
test(corrupt_type=2)'''