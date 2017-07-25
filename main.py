"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--log_name', default='name_me_', help="name of the logs output")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
parser.add_argument('--netS', default='', help="path to netS (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])
    )


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
if opt.dataset == 'mnist':
    nc = 1
    nb_label = 10
else:
    nc = 3
    nb_label = 10


#definition of generator
netG = model.netG(nz, ngf, nc)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

#definition of discriminator
netD = model.netD(ndf, nc, nb_label)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

#definition of classifier
netC = model.netC(ndf, nc, nb_label)

if opt.netC != '':
    netC.load_state_dict(torch.load(opt.netC))
print(netC)

#definition of student
netS = model.netC(ndf, nc, nb_label)

if opt.netS != '':
    netS.load_state_dict(torch.load(opt.netS))
print(netS)

#Definition of the loss functions

d_criterion = nn.BCELoss()  # Cross-entropy loss for fake/real
c_criterion = nn.CrossEntropyLoss() # Cross-Entropy for labels for classifier
s_criterion = nn.CrossEntropyLoss() # Cross-Entropy for labels for sudent


input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
d_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netC.cuda()
    netS.cuda()
    d_criterion.cuda()
    c_criterion.cuda()
    s_criterion.cuda()
    input, d_label = input.cuda(), d_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
d_label = Variable(d_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
random_label = np.random.randint(0, nb_label, opt.batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((opt.batchSize, nb_label))
random_onehot[np.arange(opt.batchSize), random_label] = 1
fixed_noise_[np.arange(opt.batchSize), :nb_label] = random_onehot[np.arange(opt.batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerC = optim.Adam(netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerS = optim.Adam(netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)



logfile_url= '%s-eval-metric-log-accuracy.txt' % ('/home/ubuntu/acgan-pytorch/output/'+opt.log_name)
print('saving logfiles  at %s' % (logfile_url))
logfile = open(logfile_url, 'a')
    #logfile.write("%s\n" % item)



for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):


        ###########################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        netC.zero_grad()
        netS.zero_grad()
        #feed input and labels from iterator
        img, label = data
        batch_size = img.size(0)
        input.data.resize_(img.size()).copy_(img)
        d_label.data.resize_(batch_size).fill_(real_label)
        c_label.data.resize_(batch_size).copy_(label)

        #forward pass on real data Discriminator/Classifier/Student

        d_output = netD(input)
        c_output = netC(input)
        s_output = netS(input)

        # compute the losses on real data for Discriminator/Classifier/Student
        d_errD_real = d_criterion(d_output, d_label)
        c_errC_real = c_criterion(c_output, c_label)
        s_errS_real = s_criterion(s_output, c_label)

        # backprop the Discriminator/Classifier/
        err_real = d_errD_real + c_errC_real
        err_real.backward()

        D_x = d_output.data.mean()
        
        #test labels and output length is the same
        correct, length = test(c_output, c_label)

        # train with fake

        #preapre batch on noise Z and uniformli sampled labels Y 
        # we might want to consider to use the same Y coming from previous dataset and soft-label from classifier for student
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
        
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
         
        # feed noise and labels in the graph
        noise.data.copy_(noise_)
        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        #forward pass generator
        fake = netG(noise)
        d_label.data.fill_(fake_label)

        #forward pass on real data Discriminator/Classifier/Student blocks gradients for generator

        d_output = netD(fake.detach())
        #c_output = netC(fake.detach()) #not using this now but might want to do knowledge distillation
        s_output = netS(fake.detach())


        # compute the losses on real data for Discriminator/Classifier/Student
        d_errD_fake = d_criterion(d_output, d_label)
        #c_err_fake = c_criterion(c_output, c_label)
        s_errS_fake = s_criterion(s_output, c_label)


        #sum all the losses --> check that is best thing to do
        err_fake = d_errD_fake + s_errS_fake 

        err_fake.backward()
        #D_G_z1 = d_output.data.mean()
        
        # update Discriminator/Classifier/Student 
        optimizerD.step()
        optimizerC.step()
        optimizerS.step()

        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        d_label.data.fill_(real_label)  # fake labels are real for generator cost

        # forward pass through Discriminator/Classifier/
        d_output = netD(fake)
        c_output = netC(fake)

        # loss function for generator
        d_errG = d_criterion(d_output, d_label)
        c_errG = c_criterion(c_output, c_label)
        errG = d_errG + c_errG

        #backward and update
        errG.backward()
        #D_G_z2 = d_output.data.mean()
        optimizerG.step()

        #summary statistics and printing 

        errD = d_errD_real + d_errD_fake
        
        if i % 50 == 0:
            
            
            
            log_output='[%d/%d][%d/%d] Loss_D: %.4f Loss_G_d: %.4f Loss_G_c: %.4f  Loss_C: %.4f Loss_S_fake: %.4f Loss_S_real: %.4f  Accuracy: %.4f / %.4f = %.4f' % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], d_errG.data[0],c_errG.data[0],c_errC_real.data[0],s_errS_fake.data[0],s_errS_real.data[0],
                 correct, length, 100.* correct / length)
            
            logfile.write(log_output+"\n")
            print(log_output)
        
        if i % 100 == 0:

            vutils.save_image(img,'%s/%s_real_samples.png' % (opt.outf,opt.log_name))
            #fake = netG(fixed_cat)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/%s_fake_samples_epoch_%03d.png' % (opt.outf,opt.log_name, epoch))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netC.state_dict(), '%s/netC_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netS.state_dict(), '%s/netS_epoch_%d.pth' % (opt.outf, epoch))

logfile.close()

