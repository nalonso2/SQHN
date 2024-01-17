import torch
from torch import nn


def to_one_hot(y_onehot, y, dev='cpu'):
    y = y.view(y_onehot.size(0), -1).to(dev)
    y_onehot.zero_().to(dev)
    y_onehot.scatter_(1, y, 1).to(dev)

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def compute_num_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct

def sigmoid_d(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def tanh_d(x):
    a = torch.tanh(x)
    return 1.0 - a ** 2.0

def relu_d(x):
    return x > 0

def piecewise(x, min=0, max=1):
    return x * (x > min).float() * (x < max).float() + (x < min).float() * min + (x > max).float() * max

def boxcar(x, min=0, max=0, beta=1):
    return beta * ((x > min).float() * (x < max).float())

def softmax_d_error(input, error):
    sm = nn.Sequential(nn.Softmax(dim=1))
    return torch.autograd.functional.vjp(sm, input, error)

def poisson(x, beta=1):
    return 1 - torch.exp(-x * beta)