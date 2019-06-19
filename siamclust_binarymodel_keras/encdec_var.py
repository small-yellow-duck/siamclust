import matplotlib
#matplotlib.use("agg")
#matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import importlib as imp
import numpy as np
import os
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim




from torch.autograd import Variable
#from tensorflow.examples.tutorials.mnist import input_data
import torchvision.datasets as dsets
from torchvision import transforms


import contrastive
import vae_net2 as net




use_cuda = False#True
print('hullo')
print('use_cuda : {use_cuda}')

mb_size = 128
lr = 1.0e-3
cnt = 0
z_dim = 24

plt.close('all')
#fig = plt.gcf()
fig = plt.figure(figsize=(4, 6))
fig.show()
fig.canvas.draw()


def makeplot(fig, samples):
    #fig = plt.figure(figsize=(4, 6))
    gs = gridspec.GridSpec(4, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    fig.canvas.draw()

def augment(X):
    #print(X.numpy().shape)
    for i in range(X.size(0)):
        r = ndimage.rotate(X[i, 0].numpy(), np.random.randint(-12, 13), reshape=False)
        #print(r.shape)
        shiftx = np.random.randint(-2,2)
        shifty = np.random.randint(-2,2)
        if shiftx > 0:
            r[shiftx:] = r[0:-shiftx]
        if shiftx < 0:
            r[0:r.shape[0]+shiftx] = r[-shiftx:]
        if shifty > 0:
            r[:, shifty:] = r[:, 0:-shifty]
        if shifty < 0:
            r[:, 0:r.shape[0]+shifty] = r[:, -shifty:]

        X[i, 0] = torch.from_numpy(r)

    return X



train = dsets.MNIST(
    root='../data/',
    train=True,
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),

    #transform = transforms.Compose([transforms.ToTensor()]),

    # transform=transforms.Compose([
    #	  transforms.ToTensor(),
    #	  transforms.Normalize((0.1307,), (0.3081,))
    # ]),
    download=True
)
test = dsets.MNIST(
    root='../data/',
    train=False,
    transform = transforms.Compose([transforms.ToTensor()])
)

train_iter = torch.utils.data.DataLoader(train, batch_size=mb_size, shuffle=True)	
val_iter = torch.utils.data.DataLoader(test, batch_size=mb_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test, batch_size=mb_size, shuffle=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=mb_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
    test,
    batch_size=mb_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=mb_size, shuffle=False, **kwargs)


KLdist = contrastive.ContrastiveDist()

KLloss = contrastive.ContrastiveLoss(margin=1.0)
#KLloss = contrastive.KL_with_sigma()
#KLloss = contrastive.KL()

enc = net.Encoder(dim=z_dim)

if use_cuda:
    enc.cuda()


def reset_grad():
    enc.zero_grad()


enc_solver = optim.RMSprop([p for p in enc.parameters()], lr=lr)
#enc_solver = optim.Adam([p for p in enc.parameters()], lr=lr)

epoch_len = 128 #4 #
max_veclen = 0.0
min_veclen = np.inf
patience = 16 #*epoch_len
patience_duration = 0
vec_len = 0.0
loss = 0.0

transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28)),  transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28))])



for it in range(400*epoch_len):
    if patience_duration > patience:
        break
    if it % epoch_len == 0:
        vec_len = 0.0

    batch_idx, (X, labels0) = next(enumerate(train_loader))

    #degrees = np.random.randint(-180, 180)
    #transform_fwd = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((degrees, degrees+1)),  transforms.ToTensor()])
    #transform_bwd = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((-degrees, -degrees+1)),  transforms.ToTensor()])

    #X_aug = augment(X)

    X = Variable(X)
    #X_aug = Variable(X_aug)

    if use_cuda:
        X = X.cuda()
        #X_aug = X_aug.cuda()
    labels = torch.zeros((mb_size, 3))
    labels[:, 0] = torch.eq(torch.cat((labels0[1:], labels0[0:1]), dim=0), torch.cat((labels0[3:], labels0[0:3]), dim=0))
    labels[:, 1] = torch.eq(torch.cat((labels0[3:], labels0[0:3]), dim=0), labels0)
    labels[:, 2] = torch.eq(labels0, torch.cat((labels0[1:], labels0[0:1]), dim=0))

    #if all three items are the same class, make the labels all 0
    labels = (torch.ones(mb_size) - torch.floor(torch.sum(labels, dim=1)/3.0)).view(-1, 1).repeat(1, 3) * labels


    # Dicriminator forward-loss-backward-update
    mu = enc(X)
    #mu_aug = enc(X_aug)
    #mu, logsigma = enc(X)

    mu1 = torch.cat((mu[1:], mu[0:1]), dim=0)
    mu2 = torch.cat((mu[3:], mu[0:3]), dim=0)

    KLdist0 = KLdist(mu1, mu2).view(-1, 1)
    KLdist1 = KLdist(mu2, mu).view(-1, 1)
    KLdist2 = KLdist(mu, mu1).view(-1, 1)

    dist = torch.cat((KLdist0, KLdist1, KLdist2), dim=1)
    dummy, argmin = torch.min(dist, dim=1)
    onehot = torch.zeros_like(dist)
    argmin = argmin.view(-1, 1).long()
    onehot.scatter_(1, argmin, 1)

    use_labels = (torch.sum(labels, dim=1)).view(-1, 1).repeat(1, 3)

    if use_cuda:
        labels = labels.cuda()
        use_labels = use_labels.cuda()

    labels = use_labels * labels + (torch.ones_like(use_labels)-use_labels)*onehot

    labels = Variable(labels)

    enc_loss = KLloss(mu1, mu2, labels[:, 0:1])
    enc_loss += KLloss(mu2, mu, labels[:, 1:2])
    enc_loss += KLloss(mu, mu1, labels[:, 2:3])

    if use_cuda:
        loss += enc_loss.data.cpu().numpy()
    else:
        loss += enc_loss.data.numpy()

    enc_loss.backward()
    enc_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    #vec_len += torch.mean(torch.sqrt(torch.mean((mu2-(torch.mean(mu2, 0)).repeat(mb_size, 1))**2, 1))).data.cpu().numpy()
    vec_len += (torch.mean(torch.pow(mu, 2))).data.cpu().numpy()





    # Print and plot every now and then
    if it % (epoch_len) == 0:
        plt.close('all')
        #print('Iter-{}; enc_loss: {}; dec_loss: {}'
        #	  .format(it, enc_loss.data.cpu().numpy(), dec_loss.data.cpu().numpy()))

        vec_len = vec_len/epoch_len
        loss = loss / epoch_len
        print('Iter-{}; enc_loss: {}; vec_len: {}, {}'
              .format(it, loss, vec_len, max_veclen))
        vec_len = 0.0
        loss = 0.0




        #if not os.path.exists('out/'):
        #    os.makedirs('out/')

        #plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1

#enc = torch.load('enc_model.pt')

