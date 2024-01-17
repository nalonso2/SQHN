import pickle
#import matplotlib.pyplot as plt
import torch
import numpy as np
from utilities import sigmoid_d
from utilities import tanh_d
import matplotlib
import pylab
import math


matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['savefig.dpi']=400.
matplotlib.rcParams['font.size']=9.0
matplotlib.rcParams['figure.figsize']=(5.0,3.5)
matplotlib.rcParams['axes.formatter.limits']=[-10,10]
matplotlib.rcParams['axes.labelsize']= 9.0
matplotlib.rcParams['figure.subplot.bottom'] = .2
matplotlib.rcParams['figure.subplot.left'] = .2
matplotlib.rcParams["axes.facecolor"] = (0.8, 0.8, 0.8, 0.5)
matplotlib.rcParams['axes.edgecolor'] = 'white'
matplotlib.rcParams['grid.linewidth'] = 1.2
matplotlib.rcParams['grid.color'] = 'white'
matplotlib.rcParams['axes.grid'] = True


def compute_means(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0])))

        for m in range(0,len(data)):
            d_tensor[m, :] = torch.tensor(data[m]).view(1,-1).clone()

        return torch.mean(d_tensor*scale, dim=0)


def compute_stds(data, scale=1):
    with torch.no_grad():
        d_tensor = torch.zeros((len(data), len(data[0])))

        for m in range(0,len(data)):
            d_tensor[m, :] = torch.tensor(data[m]).view(1,-1).clone()

        return torch.std(d_tensor*scale, dim=0)



def plot_compare(hip_sz=500, noise=0.1):


    with open(f'data/IPHNAA_numN{hip_sz}_noise{noise}.data', 'rb') as filehandle:
        iphn = pickle.load(filehandle)

    with open(f'data/MHNAA_numN{hip_sz}_noise{noise}.data', 'rb') as filehandle:
        mhn = pickle.load(filehandle)



    iphnData = [sum(iphn[0]) / len(iphn[0]), sum(iphn[1]) / len(iphn[1]), sum(iphn[2]) / len(iphn[2])]
    mhnData = [sum(mhn[0]) / len(mhn[0]), sum(mhn[1]) / len(mhn[1]), sum(mhn[2]) / len(mhn[2])]

    print('I-PHN', 'Noise:', iphnData[0], 'PixDropout', iphnData[1], 'Mask', iphnData[2])
    print('MHN', 'Noise:', mhnData[0], 'PixDropout', mhnData[1], sum(mhn[1]), 'Mask', mhnData[2], sum(mhn[2]))


    # Make the plot
    pylab.subplot(131)
    pylab.bar([1], iphnData[0], width=.4, edgecolor='grey', label='SQHN')
    pylab.bar([1.4], mhnData[0], width=.4, edgecolor='grey', label='MHN')
    pylab.title('Noise')
    pylab.ylabel('Recall Acc.')
    pylab.legend()
    pylab.xticks([])

    pylab.subplot(132)
    pylab.bar([1], iphnData[1], width=.4, edgecolor='grey')
    pylab.bar([1.4], mhnData[1], width=.4, edgecolor='grey')
    pylab.title('Pixel Dropout')
    pylab.xticks([])

    pylab.subplot(133)
    pylab.bar([1], iphnData[2], width=.4, edgecolor='grey')
    pylab.bar([1.4], mhnData[2], width=.4, edgecolor='grey')
    pylab.title('Mask')
    pylab.xticks([])

    # Adding x-ticks
    pylab.show()



def plot_capacity_tests():

    fig, axs = pylab.subplots(3, 3, figsize=(6, 6), sharey=True, sharex=True)

    #names = [['dot', 'cos'], ['max', 'softmax']]
    names = [r'MHN($\beta=1$)', r'MHN($\beta=\infty$)', 'SQHN']
    x = [100, 500, 1000, 2500, 5000, 7500]
    lst = ['-', ':', '-', ':']
    noises = [0, 0.25, [0]]
    tests = [0,0,2]
    datas = [0,2,4]
    acts = [1, 0, 0]
    mds = [0,0,1]
    fm = ['o', '^', 'o']
    colors = ['#1f77b4', 'black', 'red']


    for t_type in range(3):
        for m in range(3):
            for dt in range(3):
                with open(f'data/AutoA_Model{mds[m]}_ActF{acts[m]}_Test{tests[t_type]}_numN[100, 500, 1000, 2500, 5000, 7500]_noise{noises[t_type]}_frcMsk[0]_data{datas[dt]}.data', 'rb') as filehandle:
                    d = pickle.load(filehandle)

                    axs[t_type,dt].errorbar(x, d[0], yerr=d[1], fmt=fm[m], alpha=.6, label=names[m], ls=lst[m], markersize=6, color=colors[m])
                    axs[t_type,dt].set(ylim=[-.05, 1.1])


    axs[0,1].legend(loc='center')
    axs[2, 0].set(xlabel='# Images')
    axs[2, 1].set(xlabel='# Images')
    axs[2, 2].set(xlabel='# Images')
    axs[0,0].set(ylabel='Recall(None)')
    axs[1,0].set(ylabel='Recall(Gauss.)')
    axs[2,0].set(ylabel='Recall(Mask)')
    axs[0,0].set(title='MNIST')
    axs[0,1].set(title='CIFAR-100')
    axs[0,2].set(title='TinyImageNet')

    pylab.show()


def plot_corruption_tests(data=2):

    fig, axs = pylab.subplots(1, 5, figsize=(12, 3), sharey=True)

    if data == 2:
        title='CIFAR-100'
    else:
        title='Tiny Imagenet'

    names = ['dot', 'cos', 'l1', 'l2']
    x = [[.05, .15, .25, .4, .5, .75, 1, 1.25, 1.5], [.05, .15, .25, .4, .5, .75, 1, 1.25, 1.5],
         [.1, .25, .5, .75, 7/8, 15/16], [.1, .25, .5, .75, 7/8, 15/16], [.1, .25, .5, .75, 7/8, 15/16]]

    noise = [[.05, .15, .25, .4, .5, .75, 1, 1.25, 1.5], [.05, .15, .25, .4, .5, .75, 1, 1.25, 1.5], [0.5], [0.5], [0.5]]
    frcMsk = [[0.5], [0.5], [.1, .25, .5, .75, 7/8, 15/16], [.1, .25, .5, .75, 7/8, 15/16], [.1, .25, .5, .75, 7/8, 15/16]]
    lst = ['-', '-', '--', ':']
    t_types = [4,5,6,7,8]

    for t_type in range(5):
        for m_type in range(4):
            with open(f'data/AutoA_Model{m_type}_Test{t_types[t_type]}_numN[1000]_noise{noise[t_type-5]}_frcMsk{frcMsk[t_type-5]}_data{data}.data', 'rb') as filehandle:
                d = pickle.load(filehandle)

                axs[t_type].errorbar(x[t_type], d[0], yerr=d[1], fmt='o', alpha=.7, label=names[m_type], ls=lst[m_type], markersize=4)
                axs[t_type].set(ylim=[-.05, 1.05])

    axs[1].legend()
    axs[0].set(ylabel='Fraction Retrieved')
    axs[0].set(title='Noise')
    axs[1].set(title='Noise w/o Clamp')
    axs[2].set(title='Pixel Drop')
    axs[3].set(title='Mask')
    axs[4].set(title='Mask w/ Attention')
    axs[0].set(xlabel='Noise Variance')
    axs[1].set(xlabel='Noise Variance')
    axs[2].set(xlabel='Fraction Dropped')
    axs[3].set(xlabel='Fraction Mask')
    axs[4].set(xlabel='Fraction Mask')
    fig.suptitle(title, y=1)
    #pylab.tight_layout()
    pylab.show()

##########################################################################################################################

def plot_sample_tests():
    fig, axs = pylab.subplots(1, 5, figsize=(12, 2.5), sharey=True)

    names = ['dot', 'cos', 'l1', 'l2']
    x = [100, 500, 1000, 2500, 5000, 7500]
    lst = ['-', '-', '--', ':']

    for d in range(5):
        for m_type in range(4):
            with open(f'data/AutoA_Model{m_type}_Test3_numN[100, 500, 1000, 2500, 5000, 7500]_noise[0.5]_attnFalse_frcMsk[0.5]_data{d}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)

                axs[d].errorbar(x, dta[0], yerr=dta[1], fmt='o', alpha=.7, label=names[m_type], ls=lst[m_type], markersize=4)
                axs[d].set(ylim=[-.05, 1.05])

    axs[2].set(xlabel='# Images')
    axs[0].legend(loc='best')
    axs[0].set(ylabel='Fraction Retrieved')
    axs[0].set(title='MNIST')
    axs[1].set(title='F-MNIST')
    axs[2].set(title='CIFAR-100')
    axs[3].set(title='SVHN')
    axs[4].set(title='Tiny Imagenet')
    fig.suptitle('Retrieval w/ Binary Samples', y=1.1)
    #pylab.tight_layout()
    pylab.show()

###################################################################################################################

def plot_online_aa():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 3, figsize=(9, 3), sharey=True)

        names = ['IPHN-Grad+Mod.', 'IPHN-Simple', 'SQHN-Grad', 'MHN-BP']
        x = torch.linspace(1, 300, int(300/5))
        lst = ['-', '-', '--', ':']
        hip_szs = [10, 30, 50]
        wus = [1,2,0]

        for wu in wus:
            it = 0
            for hp_sz in hip_szs:
                with open(f'data/AA_Online_Model1_numN{hp_sz}_data0_numData300_upType{wu}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    axs[it].errorbar(x, dta[0], yerr=dta[1], fmt='o', alpha=.3, label=names[wu], markersize=2)
                    it += 1
        it = 0
        for hp_sz in hip_szs:
            with open(f'data/AA_OnlineBP_numN{hp_sz}_data0_numData300.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                axs[it].errorbar(x, dta[0], yerr=dta[1], fmt='o', alpha=.3, label=names[3], markersize=2)
                it += 1


        axs[1].set(xlabel='Training Iteration')
        axs[0].legend(loc='best')
        axs[0].set(ylabel='Retrieval MSE')
        axs[0].set(title='N=10')
        axs[1].set(title='N=30')
        axs[2].set(title='N=50')
        axs[0].set(ylim=(-0.01,.11))
        axs[1].set(ylim=(-0.01,.11))
        axs[2].set(ylim=(-0.01,.11))
        fig.suptitle('Online Auto-associative Memory', y=1.0, fontsize=12.5)
        #pylab.tight_layout()
        pylab.show()

############################################################################################################################
def plot_onlineContAbl(max_iter=3000, hid_sz=300, data=6, simf=1):
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 4, figsize=(12, 3))
        if data == 6:
            data_name = 'EMNIST'
        elif data == 4:
            data_name = 'CIFAR-100'

        names = ['SQHN', '-Dir', '-lrDecay', '-Grw+Ovr', '-Avg', '-OvWr']
        dets = [[0,3], [0,1], [3,3], [2,3], [1,3], [0,0]]
        lst = ['-', '-', '--', ':']


        for m in range(6):
            with open(f'data/AA_Online_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{dets[m][0]}_det{dets[m][1]}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))
                axs[0].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls='-', label=names[m], markersize=2.5)
                axs[2].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls='-', label=names[m], markersize=2.5)


            with open(f'data/AA_OnlineCont_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{dets[m][0]}_det{dets[m][1]}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))
                axs[1].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls='-', label=names[m], markersize=2.5)
                axs[3].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls='-', label=names[m], markersize=2.5)


        for vert in range(4):
            axs[vert].axvline(x=hid_sz, color='black', ls=':')

        axs[1].set(xlabel='Training Iteration')
        axs[0].set(xlabel='Training Iteration')
        axs[2].set(xlabel='Training Iteration')
        axs[3].set(xlabel='Training Iteration')
        axs[0].legend(loc='best', ncol=1)
        axs[2].set(ylabel='Recall MSE')
        axs[0].set(ylabel='Recall Acc.')
        axs[3].set(ylabel='Recall MSE')
        axs[1].set(ylabel='Recall Acc.')
        axs[0].set(title='Online')
        axs[1].set(title='Online+Continual')
        axs[2].set(title='Online')
        axs[3].set(title='Online+Continual')
        axs[2].set(ylim=(-0.01,.1))
        axs[3].set(ylim=(-0.01,.1))
        fig.suptitle(data_name, y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()

########################################################################################################################
def plot_onlineContAbl_Small(max_iter=3000, hid_sz=300, data=6, simf=1):
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 7, figsize=(10, 1.9), gridspec_kw={'width_ratios': [10,1,1,3,10,1,1]})
        if data == 6:
            data_name = 'EMNIST'
        elif data == 4:
            data_name = 'CIFAR-100'

        names = ['SQHN', '-Dir', '-lrDecay', '-Grw', '-Avg', '+OvWr']
        dets = [[0,0], [0,1], [3,3], [2,3], [1,3], [0,3]]
        colors = ['red', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        lst = ['-', '-', '--', ':']

        for m in range(5):
            with open(f'data/AA_OnlineCont_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{dets[m][0]}_det{dets[m][1]}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))

                axs[0].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.5, ls='-',
                                label=names[m], markersize=2.5, color=colors[m])

                axs[2].errorbar([0], torch.mean(dta[1].sum(1)/dta[1].size(1), dim=0), yerr=torch.std(dta[1].sum(1)/dta[0].size(1), dim=0), fmt='s', alpha=.5,
                                markersize=4.5, color=colors[m])

                axs[4].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.5, ls='-',
                                label=names[m], markersize=2.5, color=colors[m])

                axs[6].errorbar([0], torch.mean(dta[0].sum(1)/dta[0].size(1), dim=0), yerr=torch.std(dta[0].sum(1)/dta[0].size(1), dim=0), fmt='s',
                                alpha=.5, markersize=4.5, color=colors[m])



        axs[0].axvline(x=hid_sz, color='black', ls=':')
        axs[4].axvline(x=hid_sz, color='black', ls=':')

        axs[4].set(xlabel='Training Iteration')
        axs[0].set(xlabel='Training Iteration')
        axs[0].legend(loc='best', ncol=2)
        axs[4].set(ylabel='Recall MSE')
        axs[0].set(ylabel='Recall Acc')
        axs[6].set(ylabel='Cumulative MSE')
        axs[2].set(ylabel='Cumulative Acc')
        #axs[4].set(ylabel='Cumul. MSE')
        #axs[1].set(ylabel='Cumul. Recalled')
        axs[1].axis('off')
        axs[3].axis('off')
        axs[5].axis('off')
        axs[0].set(ylim=(-0.1, 1.1))
        axs[2].set(ylim=(-0.1, 1.1))
        axs[4].set(ylim=(-0.01, .14))
        axs[6].set(ylim=(-0.01, .14))
        axs[2].xaxis.set_visible(False)
        axs[6].xaxis.set_visible(False)
        axs[2].yaxis.set_ticklabels([])
        axs[6].yaxis.set_ticklabels([])

        #fig.suptitle(data_name)
        pylab.tight_layout()
        pylab.subplots_adjust(wspace=0)
        pylab.show()



############################################################################################################################
def plot_onlineAndCont(max_iter=3000, hid_sz=300, data=6, simf=1):
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 4, figsize=(11, 2))
        if data == 6:
            data_name = 'EMNIST'
        elif data == 4:
            data_name = 'CIFAR-100'

        names = ['MHN-SGD', 'MHN-Adam', 'SQHN']

        lst = ['-', ':']
        mrks = ['^', 'o']

        with open(f'data/AA_Online_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{0}_det{3}.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, max_iter, dta[0].size(1))
            axs[0].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')
            axs[2].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')


        with open(f'data/AA_OnlineCont_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{0}_det{3}.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, max_iter, dta[0].size(1))
            axs[1].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')
            axs[3].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')



        for o in range(2):
            with open(f'data/AA_OnlineBP_data{data}_opt{o}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))
                axs[0].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])

                axs[2].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])


            with open(f'data/AA_OnlineContBP_data{data}_opt{o}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))
                axs[1].errorbar(x, torch.mean(dta[1], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])
                axs[3].errorbar(x, torch.mean(dta[0], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])




        '''for vert in range(4):
            axs[vert].axvline(x=hid_sz, color='black', ls=':')'''

        axs[1].set(xlabel='Training Iteration')
        axs[0].set(xlabel='Training Iteration')
        axs[2].set(xlabel='Training Iteration')
        axs[3].set(xlabel='Training Iteration')
        axs[0].legend(loc='best', ncol=1)
        axs[2].set(ylabel='Recall MSE')
        axs[0].set(ylabel='Recall Acc.')
        axs[3].set(ylabel='Recall MSE')
        axs[1].set(ylabel='Recall Acc.')
        axs[0].set(title='Online')
        axs[1].set(title='Online+Continual')
        axs[2].set(title='Online')
        axs[3].set(title='Online+Continual')
        axs[2].set(ylim=(-0.01,.1))
        axs[3].set(ylim=(-0.01,.1))
        fig.suptitle(data_name, y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()


#################################################################################################################
def plot_cont(simf=1):
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 3, figsize=(8, 3))

        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        titles = ['EMNIST (OCI)', 'CIFAR-100 (OCI)', 'MNIST (ODI)']
        hid_sz = [1300, 2000]
        max_iter = [3000, 5000]
        data = [6,4]
        lst = ['-', '-', '-', '-']
        mrks = ['s', '^', 'o', 'o']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']

        for d in range(2):
            with open(f'data/AA_OnlineCont_Simf{simf}_numN{hid_sz[d]}_data{data[d]}_numData{max_iter[d]}_upType{0}_det0.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter[d], dta[0].size(1))
                axs[0,d].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-',
                                label='SQHN', markersize=2.5, color='red')

                axs[1,d].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-',
                                label='SQHN', markersize=2.5, color='red')


            for o in range(4):
                with open(f'data/AA_OnlineContBP_data{data[d]}_opt{o}_hdsz{hid_sz[d]}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    x = torch.linspace(1, max_iter[d], dta[0].size(1))
                    axs[0,d].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                    label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])

                    axs[1,d].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                    label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])



        with open(f'data/AA_OnlineContDom_Simf{simf}_numN1300_numData{3000}_upType{0}_det0.data','rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, 3000, dta[0].size(1))
            axs[0,2].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-',
                               label='SQHN', markersize=2.5, color='red')

            axs[1,2].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')

        for o in range(4):
            with open(f'data/AA_OnlineContDomBP_opt{o}_hdsz1300.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, 3000, dta[0].size(1))

                axs[0,2].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6,
                                   ls=lst[o], label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])

                axs[1,2].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6,
                                   ls=lst[o], label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])


        '''for vert in range(2):
            axs[vert,0].axvline(x=1300, color='gray', ls=':')
            axs[vert,2].axvline(x=1300, color='gray', ls=':')

        for vert in range(2):
            axs[vert,1].axvline(x=2000, color='gray', ls=':')'''

        #axs[0, 0].legend(loc='lower left', ncol=1)

        for x in range(3):
            axs[1, x].set(xlabel='Training Iteration')
            axs[0, x].xaxis.set_ticklabels([])

        for x in range(1,3):
            axs[0, x].yaxis.set_ticklabels([])
            axs[1, x].yaxis.set_ticklabels([])

        axs[1,0].set(ylabel='Recall MSE')
        axs[0,0].set(ylabel='Recall Acc')
        axs[0,0].set(title='EMNIST (OCI)')
        axs[0,1].set(title='CIFAR-100 (OCI)')
        axs[0,2].set(title='MNIST (ODI)')

        for x in range(3):
            axs[1,x].set(ylim=(-0.01, .3))
            axs[0,x].set(ylim=(-0.1, 1.1))

        #pylab.tight_layout()
        pylab.show()



############################################################################################################################
def plot_online_v_cap(max_iter=3000, hid_sz=300, data=6, simf=1):
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 4, figsize=(11, 2))
        if data == 6:
            data_name = 'EMNIST'
        elif data == 4:
            data_name = 'CIFAR-100'

        names = ['MHN-SGD', 'MHN-Adam', 'SQHN']

        lst = ['-', ':']
        mrks = ['^', 'o']


        with open(f'data/AA_Online_Simf{simf}_numN{hid_sz}_data{data}_numData{max_iter}_upType{0}_det{3}.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, max_iter, dta[0].size(1))
            for a in range(2):
                axs[a*2].errorbar(x, torch.mean(dta[1 - a], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN', markersize=2.5, color='red')


            x = x / hid_sz
            for a in range(2):
                axs[a*2+1].errorbar(x, torch.mean(dta[1 - a], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6,
                                ls='-', label='SQHN', markersize=2.5, color='red')



        for o in range(2):
            with open(f'data/AA_OnlineBP_data{data}_opt{o}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[0].size(1))
                for a in range(2):
                    axs[a*2].errorbar(x, torch.mean(dta[1-a], dim=0), yerr=torch.std(dta[1], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])
                x = x / hid_sz
                for a in range(2):
                    axs[a*2+1].errorbar(x, torch.mean(dta[1-a], dim=0), yerr=torch.std(dta[0], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color='black', marker=mrks[o])


        axs[1].set(xlabel='#Images/N')
        axs[0].set(xlabel='Training Iteration')
        axs[3].set(xlabel='#Images/N')
        axs[2].set(xlabel='Training Iteration')
        axs[0].legend(loc='best', ncol=1)
        axs[2].set(ylabel='Recall MSE')
        axs[0].set(ylabel='Recall Acc.')
        axs[3].set(ylabel='Recall MSE')
        axs[1].set(ylabel='Recall Acc.')
        axs[2].set(ylim=(-0.01,.1))
        axs[3].set(ylim=(-0.01,.1))
        fig.suptitle(data_name, y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()



############################################################################################################################
def plot_noisy_online(max_iter=300, hid_sz=300, data=6):
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 2, figsize=(5, 2.5))

        names = ['MHN-SGD', 'MHN-Adam']
        namesp = ['SQHN', 'SQHN+']
        clrs = ['#1f77b4', 'black']

        if data == 6:
            dt_nm = 'EMNIST'
        else:
            dt_nm = 'CIFAR-100'

        lst = ['-', ':']
        mrks = ['o', '^']
        num_up = [1, 5, 20, 50]

        for nst in range(2):
            for plus in range(2):
                y_mse = []
                y_rec = []
                std_mse = []
                std_rec = []
                for nup in range(4):
                    with open(f'data/AA_NoisyOnline_numN{hid_sz}_data{data}_numData{max_iter}_NUp{num_up[nup]}_NsType{nst}_plus{bool(plus)}.data', 'rb') as filehandle:
                        dta = pickle.load(filehandle)
                        y_mse.append(torch.mean(dta[0][:, -1]))
                        std_mse.append(torch.std(dta[0][:, -1]))
                        y_rec.append(torch.mean(dta[1][:, -1]))
                        std_rec.append(torch.std(dta[1][:, -1]))

                axs[0,nst].errorbar(num_up, y_rec, yerr=std_rec, fmt='o', alpha=.6, ls=lst[plus], label=namesp[plus],
                                    markersize=4.5, color='red', marker=mrks[plus])
                axs[1,nst].errorbar(num_up, y_mse, yerr=std_mse, fmt='o', alpha=.6, ls=lst[plus], label=namesp[plus],
                                    markersize=4.5, color='red', marker=mrks[plus])



        for o in range(2):
            for nst in range(2):
                y_mse = []
                y_rec = []
                std_mse = []
                std_rec = []
                for nup in range(4):
                    with open(f'data/AA_NoisyOnlineBP_data{data}_opt{o}_NUp{num_up[nup]}_NsType{nst}.data','rb') as filehandle:
                        dta = pickle.load(filehandle)
                        y_mse.append(torch.mean(dta[0][:, -1]))
                        std_mse.append(torch.std(dta[0][:, -1]))
                        y_rec.append(torch.mean(dta[1][:, -1]))
                        std_rec.append(torch.std(dta[1][:, -1]))

                axs[0,nst].errorbar(num_up, y_rec, yerr=std_rec, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o],
                                      markersize=4.5, color=clrs[o])
                axs[1,nst].errorbar(num_up, y_mse, yerr=std_mse, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o],
                                          markersize=4.5, color=clrs[o])



        axs[1,0].set(xlabel='# Samples per Iter')
        axs[1,1].set(xlabel='# Samples per Iter')

        axs[0,0].set(title='White Noise')
        axs[0,1].set(title='Binary Sample')

        axs[0, 1].legend(bbox_to_anchor=(2., 0.1), loc='lower right', ncol=1)
        axs[0,0].set(ylabel='Recall Acc.')
        axs[1,0].set(ylabel='Recall MSE')

        for x in range(2):
            axs[0, x].xaxis.set_ticklabels([])
            axs[0,x].set(ylim=(-0.08, 1.1))

        for x in range(2):
            axs[x, 1].yaxis.set_ticklabels([])
            axs[1, x].set(ylim=(-0.01, .1))

        fig.suptitle('EMNIST', y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.subplots_adjust(wspace=0.07)
        pylab.show()

#######################################################################################################
def plot_tree_aa():
    with torch.no_grad():
        fig, axs = pylab.subplots(3, 4, figsize=(7, 4), sharey=True)

        names = ['SQHN 1L', 'SQHN 2L', 'SQHN 3L']
        x = [[0,.05, .15, .25, .4, .5, .75, 1, 1.25, 1.5], [0,.1, .25, .5, .75, 7/8, 15/16],
             [0,.1, .25, .5, .75, 7/8, 15/16], [0,.1, .25, .5, .75, 7/8, 15/16]]
        tst = [4,7,8,9]
        clr = [[1., 0., 0.], [.75, 0., 0.25],[0.5, 0., 0.5]]
        data = [2,4,5]
        m_type = [2,3,4]
        mkr = ['o', 'o', 'o']
        ln = ['-', '-', '-']

        for d in range(3):
            for t in range(4):
                for n in range(3):
                    with open(f'data/AutoA_Model{m_type[n]}_ActF0_Test{tst[t]}_numN[1000]_noise{x[0]}_frcMsk{x[1]}_data{data[d]}.data', 'rb') as filehandle:
                        dta = pickle.load(filehandle)
                        axs[d,t].errorbar(x[t], dta[0].to('cpu'), yerr=dta[1].to('cpu'), fmt=mkr[n], alpha=.6, label=names[n], color=clr[n], markersize=4, ls=ln[n])

        axs[2,0].set(xlabel='Noise Variance')
        axs[2,1].set(xlabel='Fraction Occluded')
        axs[2,2].set(xlabel='Fraction Occluded')
        axs[2,3].set(xlabel='Fraction Occluded')

        for x in range(2):
            for y in range(4):
                axs[x,y].xaxis.set_ticklabels([])

        axs[0,0].set(title='Noise')
        axs[0,1].set(title='Black Occlusion')
        axs[0,2].set(title='Color Occlusion')
        axs[0,3].set(title='Noise Occlusion')
        axs[2,0].legend(loc='best', ncol=1, fontsize='8')
        axs[0,0].set(ylabel='CIFAR-100')
        axs[1,0].set(ylabel='Recall Acc\n\nTiny ImgNet')
        axs[2,0].set(ylabel='Caltech 256')

        pylab.tight_layout()
        pylab.show()





#################################################################################################
def plot_online_test(cont=False, data=2, max_iter=5001, wtup=0):
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 2, figsize=(5, 2))

        names = ['VM (75)', 'VM (200)', 'VM (500)', 'VM (Tree)']
        x = torch.linspace(1, max_iter, int(300/5))
        hid_szs = [75, 200, 500]


        for h in range(len(hid_szs)):
            with open(f'data/AA_Online_Simf1_numN{hid_szs[h]}_data{data}_numData{max_iter}_upType{wtup}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, max_iter, dta[-2].size(1))
                axs[0].errorbar(x, torch.mean(dta[-2], dim=0), yerr=torch.std(dta[-2], dim=0), ls='-', fmt='o', alpha=.3, label=names[h], markersize=2)
                axs[1].errorbar(dta[-1], torch.mean(dta[-2], dim=0)[-1], yerr=torch.std(dta[-2], dim=0)[-1], fmt='o', alpha=.3, label=names[h], markersize=10)


        with open(f'data/AA_Online_TreeArch5_data{data}_numData{max_iter}_wtupType{wtup}.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, max_iter, dta[-2].size(1))
            axs[0].errorbar(x, torch.mean(dta[-2], dim=0), yerr=torch.std(dta[-2], dim=0), ls='-', fmt='o', alpha=.3, label=names[3], markersize=2)
            axs[1].errorbar(dta[-1], torch.mean(dta[-2], dim=0)[-1], yerr=torch.std(dta[-2], dim=0)[-1], fmt='o', alpha=.3, label=names[h], markersize=10)


        axs[0].legend(loc='best')
        axs[0].set(ylabel='Test MSE')
        axs[0].set(xlabel='Training Iteration')
        axs[0].set(ylim=(0.04, .067))
        #axs[1].set(ylabel='Test MSE')
        axs[1].set(xlabel='# Non-zero Parameters')
        #axs[1].set(ylim=(0.04, .071))
        #fig.suptitle('Online Auto-associative Memory', y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()





####################################################################################################################
def plot_recog():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 4, figsize=(9, 2.3))
        names = ['SQHN', 'SQHN(mse)', 'MHN', 'MHN(mse)']
        lst = ['-', ':']

        for rec_t in range(2):
            with open(f'data/Recogn_Simf1_numN300_data0_numData3000_upType0_recT{rec_t}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)

                for z in range(3):
                    y = torch.mean(dta[-(3-z)], dim=0)
                    std = torch.std(dta[-(3-z)], dim=0)
                    x = torch.linspace(0, 3000, y.size(0))
                    axs[z].errorbar(x, y, yerr=std, fmt='o', alpha=.6, label=names[rec_t], markersize=2.5, ls='-')

                y = torch.mean(sum(dta[-3:]) / 3, dim=0)
                std = torch.std(sum(dta[-3:]) / 3, dim=0)
                x = torch.linspace(0, 3000, y.size(0))
                axs[3].errorbar(x, y, yerr=std, fmt='o', alpha=.6, label=names[rec_t], markersize=2.5, ls='-')

        g = [70, .5]
        for rec_t in range(2):
            with open(f'data/MHN_Recogn_beta0.05_numN300_data0_numData3000_gamma{g[rec_t]}_rect{rec_t}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)

                for z in range(3):
                    y = torch.mean(dta[-(3 - z)], dim=0)
                    std = torch.std(dta[-(3 - z)], dim=0)
                    x = torch.linspace(0, 3000, y.size(0))
                    axs[z].errorbar(x, y, yerr=std, fmt='o', alpha=.6, label=names[2], markersize=2.5, ls='-')
                    axs[z].axhline(y=.5, color='black', ls=':')
                    axs[z].axvline(x=100, color='black', ls=':')

                y = torch.mean(sum(dta[-3:]) / 3, dim=0)
                std = torch.std(sum(dta[-3:]) / 3, dim=0)
                x = torch.linspace(0, 3000, y.size(0))
                axs[3].errorbar(x, y, yerr=std, fmt='o', alpha=.6, label=names[2], markersize=2.5, ls='-')



        axs[3].axhline(y=.6666, color='black', ls=':')
        axs[3].axvline(x=100, color='black', ls=':')
        for z in range(4):
            axs[z].set(xlabel='Training Iteration')
            axs[z].set(ylim=(.48, 1.02))
        axs[0].legend(loc='best', ncol=1)
        axs[0].set(title='Old Images')
        axs[1].set(title='New Images (In Dist.)')
        axs[2].set(title='New Images (Out Dist.)')
        axs[3].set(title='All Images')
        axs[0].set(ylabel='Proportion Recognized')
        #fig.suptitle('Auto-associative Memory Training', y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()


############################################################################################################################
def plot_cont_cumul():
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 3, figsize=(6, 3))

        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']
        mrks = ['^', 'o']
        num_hd = [[300, 1300, 2300], [700, 2000, 3300]]
        num_d = [3000, 5000]
        dt = [6,4]

        for d in range(2):
            cuml_acc = []
            cuml_mse = []
            cuml_acc_std = []
            cuml_mse_std = []
            for hdz in range(3):
                with open(f'data/AA_OnlineCont_Simf1_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                cuml_acc.append(torch.mean(dta[3].sum(1)/dta[3].size(1), dim=0))
                cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

            axs[0, d].errorbar(num_hd[d], cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls='-', label='SQHN', markersize=4.5, color='red')
            axs[1, d].errorbar(num_hd[d], cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls='-', label='SQHN', markersize=4.5, color='red')

        cuml_acc = []
        cuml_mse = []
        cuml_acc_std = []
        cuml_mse_std = []
        for hdz in range(3):
            with open(f'data/AA_OnlineContDom_Simf1_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0.data',
                      'rb') as filehandle:
                dta = pickle.load(filehandle)
            cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
            cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
            cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
            cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

        axs[0, 2].errorbar(num_hd[0], cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls='-', label='SQHN',
                           markersize=4.5, color='red')
        axs[1, 2].errorbar(num_hd[0], cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls='-', label='SQHN',
                           markersize=4.5, color='red')



        for d in range(2):
            for o in range(4):
                cuml_acc = []
                cuml_mse = []
                cuml_acc_std = []
                cuml_mse_std = []
                for hdz in range(3):
                    with open(f'data/AA_OnlineContBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data','rb') as filehandle:
                        dta = pickle.load(filehandle)
                        cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                        cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                        cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
                        cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

                axs[0, d].errorbar(num_hd[d], cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                                   markersize=4.5, color=clrs[o])
                axs[1, d].errorbar(num_hd[d], cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                                   markersize=4.5, color=clrs[o])

        for o in range(4):
            cuml_acc = []
            cuml_mse = []
            cuml_acc_std = []
            cuml_mse_std = []
            for hdz in range(3):
                with open(f'data/AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
                    cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

            axs[0, 2].errorbar(num_hd[0], cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])
            axs[1, 2].errorbar(num_hd[0], cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])


        for x in range(3):
            axs[1,x].set(xlabel='# Hid. Neurons')

        axs[0,0].set(title='EMNIST (OCI)')
        axs[0,1].set(title='CIFAR-100 (OCI)')
        axs[0,2].set(title='MNIST (ODI)')

        axs[0,2].legend(bbox_to_anchor=(2., 0.1), loc='lower right', ncol=1)

        axs[0,0].set(ylabel='Cumulative Acc.')
        axs[1,0].set(ylabel='Cumulative MSE')

        for x in range(3):
            axs[0,x].set(ylim=(-0.08,1.05))
            axs[1,x].set(ylim=(-0.01,.17))
            axs[0, x].xaxis.set_ticklabels([])

        for x in range(1,3):
            axs[0,x].yaxis.set_ticklabels([])
            axs[1,x].yaxis.set_ticklabels([])

        pylab.tight_layout()
        pylab.show()



####################################################################################################################
def plot_recog_all():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 3, figsize=(7, 2))
        names = ['SQHN', 'SQHN(mse)', 'MHN', 'MHN(mse)']
        lst = ['-', ':']

        for rec_t in range(2):
            with open(f'data/Recogn_Simf1_numN300_data0_numData3000_upType0_recT{rec_t}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)

                y = torch.mean(sum(dta[-3:]) / 3, dim=0)
                std = torch.std(sum(dta[-3:]) / 3, dim=0)
                x = torch.linspace(0, 3000, y.size(0))
                axs[0].plot(x, y, alpha=.7, label=names[rec_t], ls=lst[rec_t], color='red', linewidth=2)
                axs[0].fill_between(x, y - std, y + std, alpha=.3, color='red')

                if rec_t == 1:
                    dta_sets = ['Train Dist.', 'In Dist.', 'Out Dist.']
                    for dt in range(3):
                        y = torch.mean(dta[dt*2], dim=0)
                        std = torch.mean(dta[dt*2+1], dim=0)
                        x = torch.linspace(0, 3000, y.size(0))
                        axs[1].plot(x, y, alpha=.7, label=dta_sets[dt], linewidth=2)
                        axs[1].fill_between(x, y - std, y + std, alpha=.3)

        g = [70, .5]
        for rec_t in range(2):
            with open(f'data/MHN_Recogn_beta0.05_numN300_data0_numData3000_gamma{g[rec_t]}_rect{rec_t}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                y = torch.mean(sum(dta[-3:]) / 3, dim=0)
                std = torch.std(sum(dta[-3:]) / 3, dim=0)
                x = torch.linspace(0, 3000, y.size(0))
                axs[0].plot(x, y, alpha=.7, label=names[rec_t + 2], ls=lst[rec_t], color='black', linewidth=2)
                axs[0].fill_between(x, y - std, y + std, alpha=.3, color='black')

                if rec_t == 1:
                    for dt in range(3):
                        y = torch.mean(dta[dt*2], dim=0)
                        std = torch.mean(dta[dt*2+1], dim=0)
                        x = torch.linspace(0, 3000, y.size(0))
                        axs[2].plot(x, y, alpha=.7, label=dta_sets[dt], linewidth=2)
                        axs[2].fill_between(x, y - std, y + std, alpha=.3)

        for pl in range(3):
            axs[pl].axvline(x=300, color='gray', ls=':')
            axs[pl].set(xlabel='Training Iteration')
        axs[0].axhline(y=.66666667, color='gray', ls=':')
        axs[0].set(ylim=(.6, 1.02))
        axs[1].set(ylim=(-.005, .1))
        axs[2].set(ylim=(-.005, .1))
        axs[0].legend(loc='best', ncol=1)
        axs[1].legend(loc='best', ncol=1)
        axs[0].set(ylabel='Proportion Recognized')
        axs[1].set(ylabel='MSE')
        axs[2].set(ylabel='MSE')
        axs[1].set(title='SQHN')
        axs[2].set(title='MHN')
        #fig.suptitle('Auto-associative Memory Training', y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.show()


################################################################################################
def plot_energies():
    with torch.no_grad():
        names = ['Train (Small)', 'Train (Large)', 'Test (Large)']
        rand = [[], []]
        ff = [[], []]
        ff_fb = [[], []]
        ff_fb_iter = [[], []]

        #Small Dataset
        with open(f'data/AA_Online_TreeEnergy_Arch2_max_iter200.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)

        rand[0].append(torch.mean(dta[0][3]))
        ff[0].append(torch.mean(dta[0][0], dim=0)[0])
        ff_fb[0].append(torch.mean(dta[0][0], dim=0)[1])
        ff_fb_iter[0].append(torch.mean(dta[0][0], dim=0)[2])

        rand[1].append(torch.std(dta[0][3]))
        ff[1].append(torch.std(dta[0][0], dim=0)[0])
        ff_fb[1].append(torch.std(dta[0][0], dim=0)[1])
        ff_fb_iter[1].append(torch.std(dta[0][0], dim=0)[2])

        #Larger Dataset
        for t in range(2):
            with open(f'data/AA_Online_TreeEnergy_Arch2_max_iter1000.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)

            rand[0].append(torch.mean(dta[t][3]))
            ff[0].append(torch.mean(dta[t][0], dim=0)[0])
            ff_fb[0].append(torch.mean(dta[t][0], dim=0)[1])
            ff_fb_iter[0].append(torch.mean(dta[t][0], dim=0)[2])

            rand[1].append(torch.std(dta[0][3]))
            ff[1].append(torch.std(dta[0][0], dim=0)[0])
            ff_fb[1].append(torch.std(dta[0][0], dim=0)[1])
            ff_fb_iter[1].append(torch.std(dta[0][0], dim=0)[2])

        x_axis = torch.tensor([0., 1., 2.])
        pylab.bar(x_axis - .3, rand[0], width=0.2, label='Rand', alpha=.75, yerr=rand[1])
        pylab.bar(x_axis - .1, ff[0], width=0.2, label='FF', color='limegreen', alpha=.75, yerr=ff[1])
        pylab.bar(x_axis + .1, ff_fb[0], width=0.2, label='1x FF+FB', color='forestgreen', alpha=.75, yerr=ff_fb[1])
        pylab.bar(x_axis + .3, ff_fb_iter[0], width=0.2, label='3x FF+FB', color='darkgreen', alpha=.75, yerr=ff_fb_iter[1])
        pylab.xticks(x_axis, names)
        pylab.legend(ncol=2)
        pylab.ylabel('Energy (E)')
        pylab.grid(False)
        pylab.show()




##############################################################################################
def plot_long_cont_acc():
    with torch.no_grad():
        fig, axs = pylab.subplots(3, 1, figsize=(8, 4))
        hid_sz = [500, 1500, 2500]
        data = [6, 4, 5]
        names = ['N=500', 'N=1500', 'N=2500']

        for d in range(3):
            for hd in range(3):
                with open(f'data/AA_Online_Simf1_numN{hid_sz[hd]}_data{data[d]}_numData8000_upType0_det0.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    x = torch.linspace(1, 8000, dta[1].size(1))
                    axs[d].errorbar(x, torch.mean(dta[1], dim=0), alpha=.2 + hd*.3, ls='-',
                                   label=f'SQHN({names[hd]})', color=(1., 0., 0.), linewidth=3)


        axs[2].set(xlabel='Training Iteration')
        for x in range(3):
            axs[x].set(ylabel='Recall Acc')
        axs[0].set(title='MNIST')
        axs[1].set(title='CIFAR-100')
        axs[2].set(title='Tiny ImageNet')
        #axs[0].legend(loc='center right', bbox_to_anchor=(2., 0.5))

        for x in range(3):
            axs[x].set(ylim=(-0.1, 1.1))
        axs[0].legend(loc='center right', bbox_to_anchor=(2., 0.5))
        for x in range(2):
            axs[x].xaxis.set_ticklabels([])

        pylab.tight_layout()
        pylab.show()





##############################################################################################
def plot_long_cont_counts():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 3, figsize=(7, 2))
        hid_sz = [500, 1500, 2500]
        data = [6, 4, 5]
        names = ['N=500', 'N=1500', 'N=2500']

        for d in range(3):
            for hd in range(3):
                with open(f'data/AA_Online_Simf1_numN{hid_sz[hd]}_data{data[d]}_numData8000_upType0_det0.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    x = torch.linspace(1, dta[-1].size(0), dta[-1].size(0))[0:300]
                    axs[d].plot(x, torch.sort(dta[-1], descending=True)[0][0:300], alpha=.2 + hd*.3, ls='-', linewidth=2.5,
                                    label=f'SQHN({names[hd]})', color=(1., 0., 0.))


        for x in range(3):
            axs[x].set(xlabel='Cluster')

        for x in range(1,3):
            axs[x].yaxis.set_ticklabels([])
            #axs[1, x].yaxis.set_ticklabels([])

        #axs[1,0].set(ylabel='Recall MSE')
        axs[0].set(ylabel='# Data points')
        axs[0].set(title='MNIST')
        axs[1].set(title='CIFAR-100')
        axs[2].set(title='Tiny ImageNet')
        axs[0].legend()

        for x in range(3):
            axs[x].set(ylim=(-0.1, 120))

        pylab.tight_layout()
        pylab.show()






####################################################################################################################
def plot_sensit():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 3, figsize=(6, 1.5))
        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']
        mrks = ['^', 'o', 'o','o']
        num_hd = [[300, 1300, 2300], [700, 2000, 3300]]
        num_d = [3000, 5000]
        dt = [6, 4]

        for d in range(2):
            sens_mse = []
            sens_mse_std = []
            for hdz in range(3):
                with open(f'data/AA_OnlineCont_Simf1_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data',
                        'rb') as filehandle:
                    dta_cont = pickle.load(filehandle)

                with open(f'data/AA_Online_Simf1_numN{num_hd[d][hdz]}_data{dt[d]}_numData{num_d[d]}_upType0_det0.data',
                        'rb') as filehandle:
                    dta_on = pickle.load(filehandle)

                cuml_cont = torch.mean(dta_cont[2])
                cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

            axs[d].plot(num_hd[d], sens_mse, marker='s', alpha=.6, ls='-', label='SQHN',
                               markersize=4.5, color='red')

        sens_mse = []
        sens_mse_std = []
        for hdz in range(3):
            with open(f'data/AA_OnlineContDom_Simf1_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0.data',
                      'rb') as filehandle:
                dta_cont = pickle.load(filehandle)

            with open(f'data/AA_OnlineContDom_Simf1_numN{num_hd[0][hdz]}_numData{num_d[0]}_upType0_det0online.data',
                      'rb') as filehandle:
                dta_on = pickle.load(filehandle)

            cuml_cont = torch.mean(dta_cont[2])
            cuml_on = torch.mean(dta_on[2])
            sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

        axs[2].errorbar(num_hd[0], sens_mse, marker='s', alpha=.6, ls='-', label='SQHN',
                           markersize=4.5, color='red')


        for d in range(2):
            for o in range(4):
                sens_mse = []
                for hdz in range(3):
                    with open(f'data/AA_OnlineContBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data', 'rb') as filehandle:
                        dta_cont = pickle.load(filehandle)
                    with open(f'data/AA_OnlineBP_data{dt[d]}_opt{o}_hdsz{num_hd[d][hdz]}.data', 'rb') as filehandle:
                        dta_on = pickle.load(filehandle)

                    cuml_cont = torch.mean(dta_cont[2])
                    cuml_on = torch.mean(dta_on[2])
                    sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

                axs[d].plot(num_hd[d], sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])


        for o in range(4):
            sens_mse = []
            for hdz in range(3):
                with open(f'data/AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}.data', 'rb') as filehandle:
                    dta_cont = pickle.load(filehandle)
                with open(f'data/AA_OnlineContDomBP_opt{o}_hdsz{num_hd[0][hdz]}online.data', 'rb') as filehandle:
                    dta_on = pickle.load(filehandle)
                cuml_cont = torch.mean(dta_cont[2])
                cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

            axs[2].plot(num_hd[0], sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])


        for x in range(3):
            axs[x].set(xlabel='# Hid. Neurons')

        axs[0].set(title='EMNIST (OCI)')
        axs[1].set(title='CIFAR-100 (OCI)')
        axs[2].set(title='MNIST (ODI)')

        axs[0].set(ylabel='Order\nSensitivity')

        for x in range(3):
            axs[x].set(ylim=(-0.01, .068))

        for x in range(1, 3):
            axs[x].yaxis.set_ticklabels([])

        pylab.tight_layout()
        pylab.show()


####################################################################################################################
def plot_sensit_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 2, figsize=(5, 1.5))
        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']
        mrks = ['^', 'o', 'o', 'o']
        num_hd = [200, 600, 1000]
        num_d = 2000
        dt = [6, 4]

        sens_mse = []
        for hdz in range(3):
            with open(f'data/AATree_OnlineCont_arch2_numN{num_hd[hdz]}_data4_numData2000_wtupType0.data',
                    'rb') as filehandle:
                dta_cont = pickle.load(filehandle)

            with open(f'data/AATree_Online_arch2_numN{num_hd[hdz]}_data4_numData2000_wtupType0.data',
                    'rb') as filehandle:
                dta_on = pickle.load(filehandle)

            cuml_cont = torch.mean(dta_cont[2])
            cuml_on = torch.mean(dta_on[2])
            sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

        axs[0].plot(num_hd, sens_mse, marker='s', alpha=.6, ls='-', label='SQHN(L3)',
                           markersize=4.5, color='purple')

        sens_mse = []
        for hdz in range(3):
            with open(f'data/AATree_OnlineContDom_arch2_numN{num_hd[hdz]}_numData2000_wtupType0.data',
                      'rb') as filehandle:
                dta_cont = pickle.load(filehandle)

            with open(f'data/AATree_OnlineContDom_arch2_numN{num_hd[hdz]}_numData2000_wtupType0online.data',
                      'rb') as filehandle:
                dta_on = pickle.load(filehandle)

            cuml_cont = torch.mean(dta_cont[2])
            cuml_on = torch.mean(dta_on[2])
            sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

        axs[1].errorbar(num_hd, sens_mse, marker='s', alpha=.6, ls='-', label='SQHN(L3)',
                           markersize=4.5, color='purple')



        for o in range(4):
            sens_mse = []
            for hdz in range(3):
                with open(f'data/AATreeBP_OnlineCont_arch1_numN{num_hd[hdz]}_data4_numData2000_optim{o}.data', 'rb') as filehandle:
                    dta_cont = pickle.load(filehandle)
                with open(f'data/AATreeBP_Online_arch1_numN{num_hd[hdz]}_data4_numData2000_optim{o}.data', 'rb') as filehandle:
                    dta_on = pickle.load(filehandle)

                cuml_cont = torch.mean(dta_cont[2])
                cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

            axs[0].plot(num_hd, sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o], markersize=4.5, color=clrs[o])


        for o in range(4):
            sens_mse = []
            for hdz in range(3):
                with open(f'data/AATreeBP_OnlineContDom_arch1_numN{num_hd[hdz]}_numData2000_optim{o}.data', 'rb') as filehandle:
                    dta_cont = pickle.load(filehandle)
                with open(f'data/AATreeBP_OnlineContDom_arch1_numN{num_hd[hdz]}_numData2000_optim{o}online.data', 'rb') as filehandle:
                    dta_on = pickle.load(filehandle)
                cuml_cont = torch.mean(dta_cont[2])
                cuml_on = torch.mean(dta_on[2])
                sens_mse.append(torch.abs(torch.mean(cuml_cont - cuml_on)))

            axs[1].plot(num_hd, sens_mse, marker='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])


        for x in range(2):
            axs[x].set(xlabel='# Hid. Neurons')

        axs[0].set(title='CIFAR-100 (OCI)')
        axs[1].set(title='CIFAR-SVHN (ODI)')

        axs[0].set(ylabel='Order\nSensitivity')

        for x in range(2):
            axs[x].set(ylim=(-0.01, .09))

        for x in range(1, 2):
            axs[x].yaxis.set_ticklabels([])

        pylab.tight_layout()
        pylab.show()


#################################################################################################################
def plot_cont_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 2, figsize=(5.1, 2.6))

        names = ['MHN-SGD', 'MHN-Adam', 'MHN-EWC++', 'MHN-ER']
        hid_sz = 600
        data = [6,4]
        lst = ['-', '-', '-', '-']
        mrks = ['s', '^', 'o', 'o']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']


        with open(f'data/AATree_OnlineCont_arch2_numN{hid_sz}_data4_numData2000_wtupType0.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, 2000, dta[0].size(1))
            axs[0,0].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN(L3)', markersize=2.5, color='purple')

            axs[1,0].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN(L3)', markersize=2.5, color='purple')


        for o in range(4):
            with open(f'data/AATreeBP_OnlineCont_arch1_numN{hid_sz}_data4_numData2000_optim{o}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, 2000, dta[0].size(1))
                axs[0,0].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])

                axs[1,0].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls=lst[o],
                                label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])



        with open(f'data/AATree_OnlineContDom_arch2_numN{hid_sz}_numData2000_wtupType0.data','rb') as filehandle:
            dta = pickle.load(filehandle)
            x = torch.linspace(1, 2000, dta[0].size(1))
            axs[0,1].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6, ls='-',
                               label='SQHN(L3)', markersize=2.5, color='purple')

            axs[1,1].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6, ls='-',
                            label='SQHN(L3)', markersize=2.5, color='purple')

        for o in range(4):
            with open(f'data/AATreeBP_OnlineContDom_arch1_numN{hid_sz}_numData2000_optim{o}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                x = torch.linspace(1, 2000, dta[0].size(1))

                axs[0,1].errorbar(x, torch.mean(dta[3], dim=0), yerr=torch.std(dta[3], dim=0), fmt='o', alpha=.6,
                                   ls=lst[o], label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])

                axs[1,1].errorbar(x, torch.mean(dta[2], dim=0), yerr=torch.std(dta[2], dim=0), fmt='o', alpha=.6,
                                   ls=lst[o], label=names[o], markersize=2.5, color=clrs[o], marker=mrks[o])



        for x in range(2):
            axs[1, x].set(xlabel='Training Iteration')
            axs[0, x].xaxis.set_ticklabels([])

        for x in range(1,2):
            axs[0, x].yaxis.set_ticklabels([])
            axs[1, x].yaxis.set_ticklabels([])

        axs[1,0].set(ylabel='Recall MSE')
        axs[0,0].set(ylabel='Recall Acc')
        axs[0,0].set(title='CIFAR-100 (OCI)')
        axs[0,1].set(title='CIFAR-SVHN (ODI)')

        for x in range(2):
            axs[1,x].set(ylim=(-0.01, .2))
            axs[0,x].set(ylim=(-0.1, 1.1))

        #pylab.tight_layout()
        pylab.show()





############################################################################################################################
def plot_cont_cumul_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 2, figsize=(5.1, 2.6))

        names = ['MHN(L3)-SGD', 'MHN(L3)-Adam', 'MHN(L3)-EWC++', 'MHN(L3)-ER']
        clrs = ['#1f77b4', 'black', 'orange', '#2ca02c']
        lst = [':', ':', ':', ':']
        mrks = ['^', 'o', 'o', 'o']
        hid_sz = [200, 600, 1000]

        for d in range(2):
            cuml_acc = []
            cuml_mse = []
            cuml_acc_std = []
            cuml_mse_std = []
            for hdz in range(3):
                with open(f'data/AATree_OnlineCont_arch2_numN{hid_sz[hdz]}_data4_numData2000_wtupType0.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                cuml_acc.append(torch.mean(dta[3].sum(1)/dta[3].size(1), dim=0))
                cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

            axs[0, 0].errorbar(hid_sz, cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls='-', label='SQHN(L3)', markersize=4.5, color='purple')
            axs[1, 0].errorbar(hid_sz, cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls='-', label='SQHN(L3)', markersize=4.5, color='purple')

        cuml_acc = []
        cuml_mse = []
        cuml_acc_std = []
        cuml_mse_std = []
        for hdz in range(3):
            with open(f'data/AATree_OnlineContDom_arch2_numN{hid_sz[hdz]}_numData2000_wtupType0.data','rb') as filehandle:
                dta = pickle.load(filehandle)
            cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
            cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
            cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
            cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

        axs[0, 1].errorbar(hid_sz, cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls='-', label='SQHN(L3)',
                           markersize=4.5, color='purple')
        axs[1, 1].errorbar(hid_sz, cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls='-', label='SQHN(L3)',
                           markersize=4.5, color='purple')




        for o in range(4):
            cuml_acc = []
            cuml_mse = []
            cuml_acc_std = []
            cuml_mse_std = []
            for hdz in range(3):
                with open(f'data/AATreeBP_OnlineCont_arch1_numN{hid_sz[hdz]}_data4_numData2000_optim{o}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
                    cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

            axs[0, 0].errorbar(hid_sz, cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])
            axs[1, 0].errorbar(hid_sz, cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])

        for o in range(4):
            cuml_acc = []
            cuml_mse = []
            cuml_acc_std = []
            cuml_mse_std = []
            for hdz in range(3):
                with open(f'data/AATreeBP_OnlineContDom_arch1_numN{hid_sz[hdz]}_numData2000_optim{o}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    cuml_mse.append(torch.mean(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_mse_std.append(torch.std(dta[2].sum(1) / dta[2].size(1), dim=0))
                    cuml_acc.append(torch.mean(dta[3].sum(1) / dta[3].size(1), dim=0))
                    cuml_acc_std.append(torch.std(dta[3].sum(1) / dta[3].size(1), dim=0))

            axs[0, 1].errorbar(hid_sz, cuml_acc, yerr=cuml_acc_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])
            axs[1, 1].errorbar(hid_sz, cuml_mse, yerr=cuml_mse_std, fmt='s', alpha=.6, ls=lst[o], label=names[o],
                               markersize=4.5, color=clrs[o])


        for x in range(2):
            axs[1,x].set(xlabel='# Hid. Neurons')

        axs[0,0].set(title='CIFAR-100 (OCI)')
        axs[0,1].set(title='CIFAR-SVHN (ODI)')

        axs[0,1].legend(bbox_to_anchor=(2., 0.1), loc='lower right', ncol=1)

        axs[0,0].set(ylabel='Cumul. Acc.')
        axs[1,0].set(ylabel='Cumul. MSE')

        for x in range(2):
            axs[0,x].set(ylim=(-0.08,1.05))
            axs[1,x].set(ylim=(-0.01,.17))
            axs[0, x].xaxis.set_ticklabels([])

        for x in range(1,2):
            axs[0,x].yaxis.set_ticklabels([])
            axs[1,x].yaxis.set_ticklabels([])

        pylab.tight_layout()
        pylab.show()


############################################################################################################################
def plot_noisyTree_online(max_iter=150, hid_sz=150, data=4):
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 2, figsize=(5, 2.5))

        names = ['MHN(L3)-SGD', 'MHN(L3)-Adam']
        namesp = ['SQHN(L3)', 'SQHN+(L3)']
        clrs = ['#1f77b4', 'black', '#2ca02c']

        lst = ['-', ':', ':']
        mrks = ['o', '^']
        num_up = [1, 10, 20]

        for nst in range(2):
            for plus in range(2):
                y_mse = []
                y_rec = []
                std_mse = []
                std_rec = []
                for nup in range(3):
                    with open(f'data/AATree_NoisyOnline_numN{hid_sz}_data{data}_numData{max_iter}_NUp{num_up[nup]}_NsType{nst}_plus{bool(plus)}.data', 'rb') as filehandle:
                        dta = pickle.load(filehandle)
                        y_mse.append(torch.mean(dta[0][:, -1]))
                        std_mse.append(torch.std(dta[0][:, -1]))
                        y_rec.append(torch.mean(dta[1][:, -1]))
                        std_rec.append(torch.std(dta[1][:, -1]))

                axs[0,nst].errorbar(num_up, y_rec, yerr=std_rec, fmt='o', alpha=.6, ls=lst[plus], label=namesp[plus],
                                    markersize=4.5, color='purple', marker=mrks[plus])
                axs[1,nst].errorbar(num_up, y_mse, yerr=std_mse, fmt='o', alpha=.6, ls=lst[plus], label=namesp[plus],
                                    markersize=4.5, color='purple', marker=mrks[plus])



        for o in range(2):
            for nst in range(2):
                y_mse = []
                y_rec = []
                std_mse = []
                std_rec = []
                for nup in range(3):
                    with open(f'data/AATree_NoisyOnlineBP_data{data}_opt{o}_NUp{num_up[nup]}_NsType{nst}.data','rb') as filehandle:
                        dta = pickle.load(filehandle)
                        y_mse.append(torch.mean(dta[0][:, -1]))
                        std_mse.append(torch.std(dta[0][:, -1]))
                        y_rec.append(torch.mean(dta[1][:, -1]))
                        std_rec.append(torch.std(dta[1][:, -1]))

                axs[0,nst].errorbar(num_up, y_rec, yerr=std_rec, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o],
                                      markersize=4.5, color=clrs[o])
                axs[1,nst].errorbar(num_up, y_mse, yerr=std_mse, fmt=mrks[o], alpha=.6, ls=lst[o], label=names[o],
                                          markersize=4.5, color=clrs[o])



        axs[1,0].set(xlabel='# Samples per Iter')
        axs[1,1].set(xlabel='# Samples per Iter')

        axs[0,0].set(title='White Noise')
        axs[0,1].set(title='Binary Sample')

        #axs[0,0].legend(loc='center right', ncol=1)
        axs[0,0].set(ylabel='Recall Acc.')
        axs[1,0].set(ylabel='Recall MSE')
        axs[0,1].legend(bbox_to_anchor=(2., 0.1), loc='lower right', ncol=1)

        for x in range(2):
            axs[0, x].xaxis.set_ticklabels([])
            axs[0,x].set(ylim=(-0.05, 1.1))

        for x in range(2):
            axs[x, 1].yaxis.set_ticklabels([])
            axs[1, x].set(ylim=(-0.01, .2))

        fig.suptitle('CIFAR-100', y=1.0, fontsize=12.5)
        pylab.tight_layout()
        pylab.subplots_adjust(wspace=0.07)
        pylab.show()


#################################################################################################################
def plot_accTest_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(2, 3, figsize=(7, 2.6))

        names = ['SQHN L1', 'SQHN L2', 'SQHN L3']
        hd_sz = [200, 600, 1000]
        nm_dt = [2500, 3000, 3500]
        archs = [9,8]
        clrs = [[1., 0, 0], [.75, 0, .25], [.5, 0., .5]]

        for sz in range(3):
            with open(f'data/AA_Online_simf2_numN{hd_sz[sz]}_data4_numData{nm_dt[sz]}_upType0_det0.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                l = dta[1].size(1) - 1
                x = torch.linspace(1, nm_dt[sz], l)

                axs[0, sz].errorbar(x, torch.mean(dta[1][:,0:l], dim=0), yerr=torch.std(dta[1][:,0:l], dim=0), fmt='o', alpha=.6,
                                   ls='-', label=names[0], markersize=2.5, color=clrs[0])

                axs[1, sz].errorbar(x, torch.mean(dta[-2][:,0:l], dim=0), yerr=torch.std(dta[-2][:,0:l], dim=0), fmt='o', alpha=.6,
                                   ls='-', label=names[0], markersize=2.5, color=clrs[0])

                lwr_bnd = [min(1.0, ((hd_sz[sz] * math.exp(- max(0., (n*100 - hd_sz[sz]) / hd_sz[sz]))) / (n*100 + .0001))) for n in range(l)]
                axs[0, sz].plot(x, lwr_bnd, alpha=.6, ls='-', label='lwr bnd', markersize=2.5, color='gray')


        for arch in range(2):
            for sz in range(3):
                with open(f'data/AATree_Online_arch{archs[arch]}_numN{hd_sz[sz]}_data4_numData{nm_dt[sz]}_wtupType0.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    l = dta[1].size(1)-1
                    x = torch.linspace(1, nm_dt[sz], l)
                    axs[0,sz].errorbar(x, torch.mean(dta[1][:,0:l], dim=0), yerr=torch.std(dta[1][:,0:l], dim=0), fmt='o', alpha=.6,
                                      ls='-', label=names[arch+1], markersize=2.5, color=clrs[arch+1])

                    axs[1,sz].errorbar(x, torch.mean(dta[-2][:,0:l], dim=0), yerr=torch.std(dta[-2][:,0:l], dim=0), fmt='o', alpha=.6,
                                      ls='-', label=names[arch+1], markersize=2.5, color=clrs[arch+1])

        for x in range(3):
            axs[1, x].set(xlabel='Training Iteration')
            axs[0, x].xaxis.set_ticklabels([])

        for x in range(1,3):
            axs[0, x].yaxis.set_ticklabels([])
            axs[1, x].yaxis.set_ticklabels([])

        axs[1,0].set(ylabel='Test MSE')
        axs[0,0].set(ylabel='Recall Acc')
        axs[0,0].set(title='#Neuron=200')
        axs[0, 1].set(title='#Neuron=600')
        axs[0, 2].set(title='#Neuron=1000')
        axs[0, 2].legend(bbox_to_anchor=(1.6, -.0), loc='lower right', ncol=1)

        for x in range(3):
            axs[1,x].set(ylim=(-0.01, .13))
            axs[0, x].set(ylim=(-0.05, 1.05))

        pylab.tight_layout()
        pylab.show()



####################################################################################################################
def plot_emerge():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 2, figsize=(6, 1.5))
        clrs = [[.66, .66, .66], [.33, .33, .33], [0,0,0]]
        lst = ['-', ':',':']
        mrks = ['^', 'o', 'o']
        num_hd = [200, 600, 1000]
        num_d = 2000
        dt = [6, 4]

        sens_mse = []
        with open(f'data/Emerge_arch8_numN1000_data5_numData2000_wtupType0.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)

            for l in range(3):
                x = torch.linspace(1, 2000, torch.mean(dta[2][:,l],dim=0).size(0))
                axs[0].errorbar(x, torch.mean(dta[2][:,l],dim=0), yerr=torch.std(dta[2][:,l],dim=0), fmt='o', alpha=.6,
                                ls='-', label=f'layer{l+1}', markersize=2.5, color=clrs[l])


                axs[1].errorbar(x, torch.mean(dta[3][:,l],dim=0), yerr=torch.std(dta[3][:,l],dim=0), fmt='o', alpha=.6,
                                ls='-', label=f'layer{l+1}', markersize=2.5, color=clrs[l])

        axs[0].set(ylabel='Neuron Growth\n(Avg. # Nrns)')
        axs[1].set(ylabel='Synapse Flexibility\n(Avg. Step Size)')
        axs[0].set(xlabel='Training Iteration')
        axs[1].set(xlabel='Training Iteration')
        axs[1].legend(bbox_to_anchor=(1.7, 0.1), loc='lower right')
        axs[1].set(ylim=(-0.01, 1.02))
        axs[0].set(ylim=(-5, 1005))
        pylab.tight_layout()
        pylab.show()




'''#################################################################################################################
def plot_recognition_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 1, figsize=(2.5, 2))

        names = ['SQHN L1', 'SQHN L2', 'SQHN L3']
        archs = [5,6]
        clrs = [[1., 0, 0], [.75, 0, .25], [.5, 0., .5]]

        with open(f'data/Recogn_Simf2_numN500_data1_numData3000_upType0_recT0.data', 'rb') as filehandle:
            dta = pickle.load(filehandle)
            y = (dta[-3] + dta[-2] + dta[-1]) / 3
            x = torch.linspace(1, 3000, y.size(1))
            axs.errorbar(x, torch.mean(y, dim=0), yerr=torch.std(y, dim=0), fmt='o', alpha=.6,
                            ls='-', label=names[0], markersize=2.5, color=clrs[0])

        for a in range(2):
            with open(f'data/RecognTree_arch{archs[a]}_data1_numData3000.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                y = (dta[-3] + dta[-2] + dta[-1]) / 3
                axs.errorbar(x, torch.mean(y, dim=0), yerr=torch.std(y, dim=0), fmt='o', alpha=.6,
                                   ls='-', label=names[a+1], markersize=2.5, color=clrs[a+1])

        axs.set(ylabel='Recognition\nAccuracy')
        axs.set(xlabel='Training Iteration')
        axs.axhline(y=.66666667, color='gray', ls=':')
        axs.axvline(x=500, color='gray', ls=':')
        pylab.tight_layout()
        pylab.show()'''

#################################################################################################################
def plot_recognition_tree():
    with torch.no_grad():
        fig, axs = pylab.subplots(1, 2, figsize=(3.5, 2))

        names = ['SQHN L1', 'SQHN L2', 'SQHN L3']
        archs = [5,6]
        clrs = [[1., 0, 0], [.75, 0, .25], [.5, 0., .5]]
        ns = ['', '_Noise(0.2)']

        for k in range(2):
            with open(f'data/Recogn_Simf2_numN500_data1_numData3000_upType0_recT0{ns[k]}.data', 'rb') as filehandle:
                dta = pickle.load(filehandle)
                y = (dta[-3] + dta[-2] + dta[-1]) / 3
                x = torch.linspace(1, 3000, y.size(1))
                axs[k].errorbar(x, torch.mean(y, dim=0), yerr=torch.std(y, dim=0), fmt='o', alpha=.6,
                                ls='-', label=names[0], markersize=2.5, color=clrs[0])

            for a in range(2):
                with open(f'data/RecognTree_arch{archs[a]}_data1_numData3000{ns[k]}.data', 'rb') as filehandle:
                    dta = pickle.load(filehandle)
                    y = (dta[-3] + dta[-2] + dta[-1]) / 3
                    axs[k].errorbar(x, torch.mean(y, dim=0), yerr=torch.std(y, dim=0), fmt='o', alpha=.6,
                                       ls='-', label=names[a+1], markersize=2.5, color=clrs[a+1])

        axs[0].set(ylabel='Recognition Acc')
        axs[0].set(title='w/o Noise')
        axs[1].set(title='w/ Noise')
        axs[1].tick_params(labelleft = False)
        for x in range(2):
            axs[x].set(xlabel='Train Iteration')
            axs[x].set(ylim=(.6, 1.1))
            axs[x].set(xlabel='Train Iteration')
            axs[x].axhline(y=.66666667, color='gray', ls=':')
            axs[x].axvline(x=500, color='gray', ls=':')
        pylab.tight_layout()
        pylab.show()