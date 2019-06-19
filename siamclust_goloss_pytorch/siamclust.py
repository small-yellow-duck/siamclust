import os
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset
import torchvision.datasets as dsets
from torchvision import transforms

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import torch.nn as nn

import importlib as imp

import contrastive
#import netr as net
import net


use_cuda = True

class Dataseq(Dataset):
    """Titanic dataset."""

    def __init__(self, X, y, idx_nolabel): #, idx_wlabel, batch_size
        """
        Args:
            data: pandas dataframe
        """
        self.X = X
        self.y = y
        #self.idx_wlabel = idx_wlabel
        self.idx_nolabel = idx_nolabel
        #self.batch_size = batch_size

    def __len__(self):
        return len(self.idx_nolabel)

    def __gettrio__(self, idx):
        # sample = {col: self.data.loc[idx, col].values for dtype in self.input_dict.keys() for col in self.input_dict[dtype]}
        #idx0 = self.use_idx[idx]
        #[idx1, idx2] = np.random.choice(np.append(self.use_idx[0:idx], self.use_idx[idx+1:]), 2, replace=False)

        #[idx0, idx1, idx2] = np.random.choice(self.idx_nolabel, 3, replace=False)

        sample0 = self.X[self.idx_nolabel[idx]].float()/256.0
        #sample1 = self.X[idx1].float()/256.0
        #sample2 = self.X[idx2].float()/256.0

        return sample0 #, sample1, sample2

    def __getitem__(self, idx):
        sample = self.X[self.idx_nolabel[idx]].float()/256.0

        return sample, idx, self.y[self.idx_nolabel[idx]]




#enc = do_train()
def do_train():
    epoch_len = 8
    n_batches = 1600
    lr = 1.0e-4
    mb_size = 128
    latent_dim = 32


    ARloss = contrastive.AlwaysRight()

    train = dsets.MNIST(
        root='../data/',
        train=True,
        # transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True
    )




    train_data = Dataseq(train.train_data, train.train_labels, np.arange(train.train_labels.size(0)))


    train_iter = torch.utils.data.DataLoader(train_data, batch_size=mb_size, shuffle=True)

    enc = net.Encoder(dim=latent_dim)
    if use_cuda:
        enc = enc.cuda()

    if use_cuda:
        mu = torch.zeros(mb_size, 3, latent_dim).cuda()
        logvar = torch.zeros(mb_size, 3, latent_dim).cuda()
    else:
        mu = torch.zeros(mb_size, 3, latent_dim)
        logvar = torch.zeros(mb_size, 3, latent_dim)


    solver = optim.RMSprop([p for p in enc.parameters()], lr=lr)

    for it in range(n_batches):
        X, idx, y = next(iter(train_iter))
        if len(set(idx)) != mb_size:
            print(len(set(idx)))
        #print(y[0:5])

        if use_cuda:
            X = Variable(X).cuda()
        else:
            X = Variable(X)

        #mu[:, 0], logvar[:, 0] = enc(T)
        #mu[:, 0] = enc.reparameterize(mu[:, 0], logvar[:, 0])
        #mu[:, 1], logvar[:, 1] = torch.cat((mu[3:, 0], mu[0:3, 0]), dim=0), torch.cat((logvar[3:, 0], logvar[0:3, 0]), dim=0)
        #mu[:, 2], logvar[:, 2] = torch.cat((mu[5:, 0], mu[0:5, 0]), dim=0), torch.cat((logvar[5:, 0], logvar[0:5, 0]), dim=0)

        mu0, logvar0 = enc(X)
        mu0a = enc.reparameterize(mu0, logvar0)
        mu0b = enc.reparameterize(mu0, logvar0)
        mu1 = torch.cat((mu0a[3:], mu0a[0:3]), dim=0)
        mu2 = torch.cat((mu0b[5:], mu0b[0:5]), dim=0)
        mu = torch.cat((mu0a.unsqueeze(1), mu1.unsqueeze(1), mu2.unsqueeze(1)), 1)


        if use_cuda:
            target = torch.zeros(mb_size, 3).cuda()
        else:
            target = torch.zeros(mb_size, 3)

        loss = ARloss(mu, target)
        loss += 1.0 / 4.0 * torch.mean(torch.pow(mu, 2))
        loss += 1.0 / 4.0 * torch.mean(torch.exp(logvar) - logvar)

        mu = torch.cat((mu0a.unsqueeze(1), mu0b.unsqueeze(1), mu2.unsqueeze(1)), 1)
        target[:, 2] = 1
        loss += 0.5*ARloss(mu, target)

        loss.backward()
        solver.step()

        enc.zero_grad()

        if (it + 1) % epoch_len == 0:
            print(it+1, loss.data.cpu().numpy(), torch.mean(torch.pow(mu0, 2)).data.cpu().numpy())

    return enc


def make_preds(enc):
    mb_size = 128

    test = dsets.MNIST(
        root='../data/',
        train=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_data = Dataseq(test.train_data, test.train_labels, np.arange(test.train_labels.size(0)))
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=mb_size, shuffle=False)

    mu = torch.zeros(len(test.train_labels), enc.dim)
    y_test = torch.zeros(len(test.train_labels))
    s = 0
    for X, idx, y in test_iter:
        e = s + X.size(0)
        y_test[s:e] = y

        if use_cuda:
            X = Variable(X).cuda()
        else:
            X = Variable(X)

        mu[s:e], _ = enc(X)

        s = e

    mu = mu.data.cpu().numpy()

    pca = PCA(n_components=3, svd_solver='arpack', copy=True, whiten=True)
    pca.fit(mu[:, :])

    pca_vecs = pca.transform(mu[:, :])

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=pca_vecs[np.where(y_test==1)][:, 0], ys=pca_vecs[np.where(y_test==1)][:, 1], zs=pca_vecs[np.where(y_test==1)][:, 2], zdir='z', s=5, c='k', depthshade=True, label='1')
    ax.scatter(xs=pca_vecs[np.where(y_test==7)][:, 0], ys=pca_vecs[np.where(y_test==7)][:, 1], zs=pca_vecs[np.where(y_test==7)][:, 2], zdir='z', s=5, c='r', depthshade=True, label='7')
    ax.scatter(xs=pca_vecs[np.where(y_test==4)][:, 0], ys=pca_vecs[np.where(y_test==4)][:, 1], zs=pca_vecs[np.where(y_test==4)][:, 2], zdir='z', s=5, c='b', depthshade=True, label='4')
    ax.scatter(xs=pca_vecs[np.where(y_test==9)][:, 0], ys=pca_vecs[np.where(y_test==9)][:, 1], zs=pca_vecs[np.where(y_test==9)][:, 2], zdir='z', s=5, c='g', depthshade=True, label='9')
    ax.scatter(xs=pca_vecs[np.where(y_test==0)][:, 0], ys=pca_vecs[np.where(y_test==0)][:, 1], zs=pca_vecs[np.where(y_test==0)][:, 2], zdir='z', s=5, c='m', depthshade=True, label='0')
    plt.legend()

    tsne_vecs = TSNE(n_components=2).fit_transform(mu)
    plt.close('all')
    plt.plot(tsne_vecs[np.where(y_test==1)][:, 0], tsne_vecs[np.where(y_test==1)][:, 1], 'k.', label='1')
    plt.plot(tsne_vecs[np.where(y_test == 7)][:, 0], tsne_vecs[np.where(y_test == 7)][:, 1], 'r.', label='7')
    plt.plot(tsne_vecs[np.where(y_test == 4)][:, 0], tsne_vecs[np.where(y_test == 4)][:, 1], 'b.', label='4')
    plt.plot(tsne_vecs[np.where(y_test == 9)][:, 0], tsne_vecs[np.where(y_test == 9)][:, 1], 'g.', label='9')
    plt.plot(tsne_vecs[np.where(y_test == 0)][:, 0], tsne_vecs[np.where(y_test == 0)][:, 1], 'm.', label='0')
    plt.plot(tsne_vecs[np.where(y_test == 2)][:, 0], tsne_vecs[np.where(y_test == 2)][:, 1], 'k+', label='2')
    plt.plot(tsne_vecs[np.where(y_test == 5)][:, 0], tsne_vecs[np.where(y_test == 5)][:, 1], 'r+', label='5')
    plt.plot(tsne_vecs[np.where(y_test == 3)][:, 0], tsne_vecs[np.where(y_test == 3)][:, 1], 'b+', label='3')
    plt.plot(tsne_vecs[np.where(y_test == 8)][:, 0], tsne_vecs[np.where(y_test == 8)][:, 1], 'g+', label='8')
    plt.legend()


    '''
    ii = 0
    jj = 5
    kk = 9
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=mu[np.where(y_test==1)][:, ii], ys=mu[np.where(y_test==1)][:, jj], zs=mu[np.where(y_test==1)][:, kk], zdir='z', s=5, c='k', depthshade=True, label='1')
    ax.scatter(xs=mu[np.where(y_test==7)][:, ii], ys=mu[np.where(y_test==7)][:, jj], zs=mu[np.where(y_test==7)][:, kk], zdir='z', s=5, c='r', depthshade=True, label='7')
    ax.scatter(xs=mu[np.where(y_test==4)][:, ii], ys=mu[np.where(y_test==4)][:, jj], zs=mu[np.where(y_test==4)][:, kk], zdir='z', s=5, c='b', depthshade=True, label='4')
    ax.scatter(xs=mu[np.where(y_test==9)][:, ii], ys=mu[np.where(y_test==9)][:, jj], zs=mu[np.where(y_test==9)][:, kk], zdir='z', s=5, c='g', depthshade=True, label='9')
    ax.scatter(xs=mu[np.where(y_test==0)][:, ii], ys=mu[np.where(y_test==0)][:, jj], zs=mu[np.where(y_test==0)][:, kk], zdir='z', s=5, c='m', depthshade=True, label='0')
    plt.legend()
    '''