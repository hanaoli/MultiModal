import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import pickle
import itertools
import time
import math

import numpy as np
from itertools import cycle

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.optim as optim

import pygrid
from utils.plot import text_to_pil
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from datasets.SVHNMNISTDataset import SVHNMNIST

class ClfImg(nn.Module):
    def __init__(self):
        super(ClfImg, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2);
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2);
        self.relu = nn.ReLU();
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        return h;

class ClfImgSVHN(nn.Module):
    def __init__(self):
        super(ClfImgSVHN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn1 = nn.BatchNorm2d(32);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn2 = nn.BatchNorm2d(64);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.bn3 = nn.BatchNorm2d(64);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.bn4 = nn.BatchNorm2d(128);
        self.relu = nn.ReLU();
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
        self.sigmoid = nn.Sigmoid();

    def forward(self, x):
        h = self.conv1(x);
        h = self.dropout(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.dropout(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.dropout(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.dropout(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;

    def get_activations(self, x):
        h = self.conv1(x);
        h = self.dropout(h);
        h = self.bn1(h);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.dropout(h);
        h = self.bn2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.dropout(h);
        h = self.bn3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.dropout(h);
        h = self.bn4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        return h;
##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')

    parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'celeba', 'celeba_crop', 'celeba32_sri', 'celeba64_sri', 'celeba64_sri_crop'])
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3)

    parser.add_argument('--nez', default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', default=64, help='feature dimensions of generator')
    parser.add_argument('--ndf', default=200, help='feature dimensions of ebm')

    parser.add_argument('--e_prior_sig', type=float, default=1, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=1, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='gelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=150, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')

    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_l_steps', type=int, default=30, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    parser.add_argument('--e_lr', default=0.00001, type=float)
    parser.add_argument('--g_lrs', default=0.00005, type=float)
    parser.add_argument('--g_lrm', default=0.00005, type=float)
    parser.add_argument('--g_lrt', default=0.00005, type=float)

    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--e_decay', default=0, help='weight decay for ebm')
    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')

    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--g_gamma', default=0.998, help='lr decay for gen')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--e_beta1', default=0.5, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for') # TODO(nijkamp): set to >100
    # parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=500, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=10, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')
    # parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')

    parser.add_argument('--n_fid_samples', type=int, default=50000) # TODO(nijkamp): we used 40,000 in short-run inference
    # parser.add_argument('--n_fid_samples', type=int, default=1000)

    return parser.parse_args()


def create_args_grid():
    # TODO add your enumeration of parameters here

    e_lr = [0.00002]
    e_l_step_size = [0.4]
    e_init_sig = [1.0]
    e_l_steps = [30,50,60]
    e_activation = ['lrelu']

    g_llhd_sigma = [0.3]
    g_lr = [0.0001]
    g_l_steps = [20]
    g_activation = ['lrelu']

    ngf = [64]
    ndf = [200]

    args_list = [e_lr, e_l_step_size, e_init_sig, e_l_steps, e_activation, g_llhd_sigma, g_lr, g_l_steps, g_activation, ngf, ndf]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'e_lr': args[0],
            'e_l_step_size': args[1],
            'e_init_sig': args[2],
            'e_l_steps': args[3],
            'e_activation': args[4],
            'g_llhd_sigma': args[5],
            'g_lr': args[6],
            'g_l_steps': args[7],
            'g_activation': args[8],
            'ngf': args[9],
            'ndf': args[10],
        }
        # TODO add your result metric here
        opt_result = {'fid_best': 0.0, 'fid': 0.0, 'mse': 0.0}

        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['fid_best'] = job_stats['fid_best']
    job_opt['fid'] = job_stats['fid']
    job_opt['mse'] = job_stats['mse']


##########################################################################################################
## Data

##########################################################################################################
## Model

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

def get_activation(name, args):
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(args.e_activation_leak), 'mish': Mish(), 'swish': Swish()}[name]


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class _netGS(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf*4, 4, 1, 0, bias = not args.g_batchnorm),
       #     nn.BatchNorm2d(args.ngf*8) if args.g_batchnorm else nn.Identity(),
            nn.ReLU(),

            nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias = not args.g_batchnorm),
     #       nn.BatchNorm2d(args.ngf*4) if args.g_batchnorm else nn.Identity(),
            nn.ReLU(),

            nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
       #     nn.BatchNorm2d(args.ngf*2) if args.g_batchnorm else nn.Identity(),
            nn.ReLU(),

            #nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
            #nn.BatchNorm2d(args.ngf*1) if args.g_batchnorm else nn.Identity(),
            #f,

            nn.ConvTranspose2d(args.ngf*1, args.nc, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.gen(z)

class _netGM(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)

        self.gen = nn.Sequential(
            nn.Linear(100, 400),
         #   nn.ConvTranspose2d(args.nz, args.ngf*4, 4, 1, 0, bias = not args.g_batchnorm),
          #  nn.BatchNorm2d(args.ngf*8) if args.g_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(400, 400),
         #   nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 3, 2, 1, bias = not args.g_batchnorm),
         #   nn.BatchNorm2d(args.ngf*4) if args.g_batchnorm else nn.Identity(),

            nn.ReLU(),
       #     nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
        #    nn.BatchNorm2d(args.ngf*2) if args.g_batchnorm else nn.Identity(),
        #    nn.ReLU(),

            #nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
            #nn.BatchNorm2d(args.ngf*1) if args.g_batchnorm else nn.Identity(),
            #f,
            nn.Linear(400, 28*28),
        #    nn.ConvTranspose2d(args.ngf*1, 1, 4, 2, 1),
            nn.Sigmoid()
        )


    def forward(self, z):
        return self.gen(z.reshape(-1, 100)).reshape(-1, 1, 28, 28)

class _netGT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args;

        self.linear = nn.Linear(100, 128)
        self.conv1 = nn.ConvTranspose1d(128, 128,
                                        kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = nn.ConvTranspose1d(128, 128,
                                        kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv_last = nn.Conv1d(128, 71, kernel_size=1)
        self.relu = nn.ReLU()
        self.out_act = nn.Softmax(dim=-2)

    def forward(self, z):
        z = self.linear(z.view(-1, 100))
        x_hat = z.view(z.size(0), z.size(1), 1)
        x_hat = self.conv1(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv_last(x_hat)
        log_prob = self.out_act(x_hat)
        log_prob = log_prob.transpose(-2,-1)
        return log_prob

class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        apply_sn = sn if args.e_sn else lambda x: x

        f = get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.nz, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)


##########################################################################################################


def train(output_dir_job, output_dir, return_dict):

    #################################################
    ## preamble

    args = parse_args()

    set_seed(args.seed)
    makedirs_exp(output_dir)

    #################################################
    ## data

    print('Loading MNIST-SVHN-Text dataset...')
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    num_features = len(alphabet)


    train_loader = DataLoader(svhnmnist_train, batch_size=args.batch_size, shuffle=True,
                   num_workers=8, drop_last=True)
    test_loader = DataLoader(svhnmnist_test, batch_size=args.batch_size, shuffle=True,
                   num_workers=8, drop_last=True)
    def plot(p, x):
        return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(np.sqrt(args.batch_size)))

    #################################################
    ## model

    netGS = _netGS(args)
    netGM = _netGM(args)
    netGT = _netGT(args)
    netE = _netE(args)

    netGS.apply(weights_init_xavier)
    netGM.apply(weights_init_xavier)
    netGT.apply(weights_init_xavier)
    netE.apply(weights_init_xavier)

    netGS = netGS.cuda()
    netGM = netGM.cuda()
    netGT = netGT.cuda()
    netE = netE.cuda()

    def eval_flag():
        netGS.eval()
        netGM.eval()
        netGT.eval()
        netE.eval()

    def train_flag():
        netGS.train()
        netGM.train()
        netGT.train()
        netE.train()

    def energy(score):
        if args.e_energy_form == 'tanh':
            energy = F.tanh(-score.squeeze())
        elif args.e_energy_form == 'sigmoid':
            energy = F.sigmoid(score.squeeze())
        elif args.e_energy_form == 'identity':
            energy = score.squeeze()
        elif args.e_energy_form == 'softplus':
            energy = F.softplus(score.squeeze())
        return energy

    mse = nn.MSELoss(reduction='sum')

    #################################################
    ## optimizer

    optE = torch.optim.Adam(netE.parameters(), lr=args.e_lr, weight_decay=args.e_decay, betas=(args.e_beta1, args.e_beta2))
    optGS = torch.optim.Adam(netGS.parameters(), lr=args.g_lrs, weight_decay=args.g_decay, betas=(args.g_beta1, args.g_beta2))
    optGM = torch.optim.Adam(netGM.parameters(), lr=args.g_lrm, weight_decay=args.g_decay, betas=(args.g_beta1, args.g_beta2))
    optGT = torch.optim.Adam(netGT.parameters(), lr=args.g_lrt, weight_decay=args.g_decay,
                             betas=(args.g_beta1, args.g_beta2))
    lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lr_scheduleGS = torch.optim.lr_scheduler.ExponentialLR(optGS, args.g_gamma)
    lr_scheduleGM = torch.optim.lr_scheduler.ExponentialLR(optGM, args.g_gamma)
    lr_scheduleGT = torch.optim.lr_scheduler.ExponentialLR(optGT, args.g_gamma)
    #################################################
    ## sampling

    def sample_p_0(n=args.batch_size, sig=args.e_init_sig):
        return sig * torch.randn(*[n, args.nz, 1, 1]).cuda()

    def sample_langevin_prior_z(z, netE, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.e_l_steps):
            en = energy(netE(z))
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * args.e_l_step_size * args.e_l_step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.e_l_with_noise:
                z.data += args.e_l_step_size * torch.randn_like(z).data

   #         if (i % 5 == 0 or i == args.e_l_steps - 1) and verbose:
   #             logger.info('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, args.e_l_steps, en.sum().item()))

            z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def sample_langevin_post_z(z, xs, xm, xt, netGS, netGM, netGT, netE, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True

        for i in range(args.g_l_steps):
            x_hat_s = netGS(z)
            x_hat_m = netGM(z)
            x_hat_t = netGT(z)
            if xs is None:
                g_log_lkhd_s = 0
                z_grad_gs = 0
                z_grad_g_grad_norm_s = None
            else:
                g_log_lkhd_s = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat_s, xs)
                z_grad_gs = torch.autograd.grad(g_log_lkhd_s, z)[0]
                z_grad_g_grad_norm_s = z_grad_gs.view(args.batch_size, -1).norm(dim=1).mean()
            if xm is None:
                g_log_lkhd_m = 0
                z_grad_gm = 0
                z_grad_g_grad_norm_m = None
            else:
                g_log_lkhd_m = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat_m, xm)
                z_grad_gm = torch.autograd.grad(g_log_lkhd_m, z)[0]
                z_grad_g_grad_norm_m = z_grad_gm.view(args.batch_size, -1).norm(dim=1).mean()
            if xt is None:
                g_log_lkhd_t = 0
                z_grad_gt = 0
                z_grad_g_grad_norm_t = None
            else:
                g_log_lkhd_t = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat_t, xt)
                z_grad_gt = torch.autograd.grad(g_log_lkhd_t, z)[0]
                z_grad_g_grad_norm_t = z_grad_gt.view(args.batch_size, -1).norm(dim=1).mean()

            en = energy(netE(z))
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]
            z_grad_e_grad_norm = z_grad_e.view(args.batch_size, -1).norm(dim=1).mean()

            z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (
                    z_grad_gs + z_grad_gm + z_grad_gt + z_grad_e + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.g_l_with_noise:
                z.data += args.g_l_step_size * torch.randn_like(z).data

        return z.detach(), z_grad_g_grad_norm_s, z_grad_g_grad_norm_m, z_grad_g_grad_norm_t, z_grad_e_grad_norm

    #################################################
    ## fid

    def get_fid(n):

        assert n <= ds_fid.shape[0]

        logger.info('computing fid with {} samples'.format(n))

        try:
            eval_flag()

            def sample_x():
                z_0 = sample_p_0().to(device)
                z_k = sample_langevin_prior_z(Variable(z_0), netE)[0]
                x_samples = to_range_0_1(netG(z_k)).clamp(min=0., max=1.).detach().cpu()
                return x_samples
            x_samples = torch.cat([sample_x() for _ in range(int(n / args.batch_size))]).numpy()
            fid = compute_fid_nchw(args, ds_fid, x_samples)
            return fid

        except Exception as e:
            print(e)
            logger.critical(e, exc_info=True)
            logger.info('FID failed')

        finally:
            train_flag()


    #################################################
    ## train

    train_flag()
    img_size = torch.Size((3, 28, 28));

    fid = 0.0
    fid_best = math.inf

    z_fixed = sample_p_0()
    x_fixed = next(iter(train_loader))
    x_fixed_m = x_fixed[0].cuda()
    x_fixed_s = x_fixed[1].cuda()
    x_fixed_t = x_fixed[2].cuda()

    stats = {
        'loss_g_s':[],
        'loss_g_m':[],
        'loss_g_t': [],
        'loss_e':[],
        'en_neg':[],
        'en_pos':[],
        'grad_norm_g_s':[],
        'grad_norm_g_m': [],
        'grad_norm_g_t': [],
        'grad_norm_e':[],
        'z_e_grad_norm':[],
        'z_g_grad_norm_s':[],
        'z_g_grad_norm_m':[],
        'z_g_grad_norm_t': [],
        'z_e_k_grad_norm':[],
        'fid':[],
    }
    interval = []

    min_fid = 1000

    clf_mnist = ClfImg().cuda()
    clf_svhn = ClfImgSVHN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_mnist = optim.Adam(clf_mnist.parameters(), lr=0.001)
    optimizer_svhn = optim.Adam(clf_svhn.parameters(), lr=0.001)

    load_clf = True

    if load_clf:
        clf_mnist = ClfImg().cuda()
        clf_mnist.load_state_dict(torch.load('clf_m1'))

        clf_svhn = ClfImgSVHN().cuda()
        clf_svhn.load_state_dict(torch.load('clf_m2'))

    else:
        for e_t in range(5):
            for i, x in enumerate(test_loader):
                xm, xs, xt, xl = x;
                xm = xm.cuda()
                xs = xs.cuda()
                xt = xt.cuda()
                xl = xl.cuda()

                optimizer_mnist.zero_grad()
                optimizer_svhn.zero_grad()

                ym = clf_mnist(xm)
                ys = clf_svhn(xs)

                loss_m = criterion(ym, xl)
                loss_s = criterion(ys, xl)

                top_pred_m = ym.argmax(1, keepdim=True)
                top_pred_s = ys.argmax(1, keepdim=True)

                correct_m = top_pred_m.eq(xl.view_as(top_pred_m)).sum()
                correct_s = top_pred_s.eq(xl.view_as(top_pred_s)).sum()

                acc_m = correct_m.float() / xl.shape[0]
                acc_s = correct_s.float() / xl.shape[0]

                loss_m.backward()
                loss_s.backward()
                optimizer_mnist.step()
                optimizer_svhn.step()
            print('Acc Mnist', acc_m)
            print('Acc Svhn', acc_s)

        save_dict = {
            'CLF_MNIST': clf_mnist.state_dict(),
            'CLF_SVHN': clf_svhn.state_dict(),
        }
        torch.save(save_dict, '{}/ckpt/ckpt_.pth'.format(output_dir))

    cross_mnist = 0
    max_cross_mnist = 0
    cross_svhn = 0
    max_cross_svhn = 0
    joint_co = 0
    max_joint_co = 0

    for epoch in range(args.n_epochs):
        for i, x in enumerate(train_loader):
            xm, xs, xt, xl = x;
            xm = xm.cuda()
            xs = xs.cuda()
            xt = xt.cuda()
            xl = xl.cuda()
            train_flag()

            batch_size = xm.shape[0]

            # Initialize chains
            z_g_0 = sample_p_0(n=batch_size)
            z_e_0 = sample_p_0(n=batch_size)

            # Langevin posterior and prior
            z_g_k, z_g_grad_norm_s, z_g_grad_norm_m, z_g_grad_norm_t, z_e_grad_norm = sample_langevin_post_z(Variable(z_g_0), xs, xm, xt,
                                                                                            netGS,
                                                                                            netGM, netGT, netE,
                                                                                            verbose=(i == 0))
            z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE, verbose=(i == 0))

            # Learn generator
            optGS.zero_grad()
            x_hat_s = netGS(z_g_k.detach())
            loss_g_s = mse(x_hat_s, xs) / batch_size
            loss_g_s.backward()
            grad_norm_g_s = get_grad_norm(netGS.parameters())
            if args.g_is_grad_clamp:
                torch.nn.utils.clip_grad_norm(netGS.parameters(), opt.g_max_norm)
            optGS.step()

            optGM.zero_grad()
            x_hat_m = netGM(z_g_k.detach())
            loss_g_m = mse(x_hat_m, xm) / batch_size
            loss_g_m.backward()
            grad_norm_g_m = get_grad_norm(netGM.parameters())
            if args.g_is_grad_clamp:
                torch.nn.utils.clip_grad_norm(netGM.parameters(), opt.g_max_norm)
            optGM.step()

            optGT.zero_grad()
            x_hat_t = netGT(z_g_k.detach())
            loss_g_t = mse(x_hat_t, xt) / batch_size
            loss_g_t.backward()
            grad_norm_g_t = get_grad_norm(netGT.parameters())
            if args.g_is_grad_clamp:
                torch.nn.utils.clip_grad_norm(netGT.parameters(), opt.g_max_norm)
            optGT.step()

            # Learn prior EBM
            optE.zero_grad()
            en_neg = energy(netE(
                z_e_k.detach())).mean()  # TODO(nijkamp): why mean() here and in Langevin sum() over energy? constant is absorbed into Adam adaptive lr
            en_pos = energy(netE(z_g_k.detach())).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            grad_norm_e = get_grad_norm(netE.parameters())
            if args.e_is_grad_clamp:
                torch.nn.utils.clip_grad_norm_(netE.parameters(), args.e_max_norm)
            optE.step()

            # Printout
            if i % args.n_printout == 0:
                with torch.no_grad():
                    x_0_s = netGS(z_e_0)
                    x_k_s = netGS(z_e_k)

                    x_0_m = netGM(z_e_0)
                    x_k_m = netGM(z_e_k)

                    x_0_t = netGT(z_e_0)
                    x_k_t = netGT(z_e_k)
                    en_neg_2 = energy(netE(z_e_k)).mean()
                    en_pos_2 = energy(netE(z_g_k)).mean()

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(),
                                                                             z_g_k.abs().max())

                    print('{:5d}/{:5d} {:5d}/{:5d} '.format(epoch, args.n_epochs, i, len(train_loader)) +
                          'loss_g_s={:8.3f}, '.format(loss_g_s) +
                          'loss_g_m={:8.3f}, '.format(loss_g_m) +
                          'loss_g_t={:8.3f}, '.format(loss_g_t) +
                          'loss_e={:8.3f}, '.format(loss_e) +
                          'en_pos=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_pos, en_pos_2, en_pos_2 - en_pos) +
                          'en_neg=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_neg, en_neg_2, en_neg_2 - en_neg) +
                          '|grad_g_s|={:8.2f}, '.format(grad_norm_g_s) +
                          '|grad_g_m|={:8.2f}, '.format(grad_norm_g_m) +
                          '|grad_g_t|={:8.2f}, '.format(grad_norm_g_t) +
                          '|grad_e|={:8.2f}, '.format(grad_norm_e) +
                          '|z_g_grad_s|={:7.3f}, '.format(z_g_grad_norm_s) +
                          '|z_g_grad_m|={:7.3f}, '.format(z_g_grad_norm_m) +
                          '|z_g_grad_t|={:7.3f}, '.format(z_g_grad_norm_t) +
                          '|z_e_grad|={:7.3f}, '.format(z_e_grad_norm) +
                          '|z_e_k_grad|={:7.3f}, '.format(z_e_k_grad_norm) +
                          '|z_g_0|={:6.2f}, '.format(z_g_0.view(batch_size, -1).norm(dim=1).mean()) +
                          '|z_g_k|={:6.2f}, '.format(z_g_k.view(batch_size, -1).norm(dim=1).mean()) +
                          '|z_e_0|={:6.2f}, '.format(z_e_0.view(batch_size, -1).norm(dim=1).mean()) +
                          '|z_e_k|={:6.2f}, '.format(z_e_k.view(batch_size, -1).norm(dim=1).mean()) +
                          'z_e_disp={:6.2f}, '.format((z_e_k - z_e_0).view(batch_size, -1).norm(dim=1).mean()) +
                          'z_g_disp={:6.2f}, '.format((z_g_k - z_g_0).view(batch_size, -1).norm(dim=1).mean()) +
                          'x_e_disp_s={:6.2f}, '.format((x_k_s - x_0_s).view(batch_size, -1).norm(dim=1).mean()) +
                          'x_e_disp_m={:6.2f}, '.format((x_k_m - x_0_m).view(batch_size, -1).norm(dim=1).mean()) +
                          'x_e_disp_t={:6.2f}, '.format((x_k_t - x_0_t).reshape(batch_size, -1).norm(dim=1).mean()) +
                          'prior_moments={}, '.format(prior_moments) +
                          'posterior_moments={}, '.format(posterior_moments) +
                          'min_fid={}, '.format(min_fid) +
                          'cross_mnist={}, '.format(max_cross_mnist) +
                          'cross_svhn={}, '.format(max_cross_svhn) +
                          'join_co={}, '.format(max_joint_co)
                          )

            if i % (5 * args.n_printout) == 0:
                batch_size_fixed = x_fixed_s.shape[0]

                z_g_0 = sample_p_0(n=batch_size_fixed)
                z_e_0 = sample_p_0(n=batch_size_fixed)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), x_fixed_s, None, None, netGS, netGM, netGT,
                                                           netE)
                z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE)
                tmp = torch.zeros((x_fixed_s.shape[0], 3, 28, 28))
                with torch.no_grad():
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_pos_fixed.png'.format(output_dir, epoch, i), x_fixed_s)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_pos_s.png'.format(output_dir, epoch, i), netGS(z_g_k))
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_pos_m.png'.format(output_dir, epoch, i), netGM(z_g_k))
                    gen_text = netGT(z_g_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j+1], img_size, alphabet)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_pos_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_0_s.png'.format(output_dir, epoch, i), netGS(z_e_0))
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_0_m.png'.format(output_dir, epoch, i), netGM(z_e_0))
                    gen_text = netGT(z_e_0)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_0_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_k_s.png'.format(output_dir, epoch, i), netGS(z_e_k))
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_k_m.png'.format(output_dir, epoch, i), netGM(z_e_k))
                    gen_text = netGT(z_e_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_neg_k_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_fixed_s.png'.format(output_dir, epoch, i), netGS(z_fixed))
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_fixed_m.png'.format(output_dir, epoch, i), netGM(z_fixed))
                    gen_text = netGT(z_fixed)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_fixed_t.png'.format(output_dir, epoch, i), tmp)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), None, x_fixed_m, None, netGS, netGM, netGT,
                                                           netE)
                z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE)

                with torch.no_grad():
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_pos_fixed.png'.format(output_dir, epoch, i), x_fixed_m)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_pos_s.png'.format(output_dir, epoch, i), netGS(z_g_k))
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_pos_m.png'.format(output_dir, epoch, i), netGM(z_g_k))
                    gen_text = netGT(z_g_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_pos_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_0_s.png'.format(output_dir, epoch, i), netGS(z_e_0))
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_0_m.png'.format(output_dir, epoch, i), netGM(z_e_0))
                    gen_text = netGT(z_e_0)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_0_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_k_s.png'.format(output_dir, epoch, i), netGS(z_e_k))
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_k_m.png'.format(output_dir, epoch, i), netGM(z_e_k))
                    gen_text = netGT(z_e_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_neg_k_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_fixed_s.png'.format(output_dir, epoch, i),
                         netGS(z_fixed))
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_fixed_m.png'.format(output_dir, epoch, i),
                         netGM(z_fixed))
                    gen_text = netGT(z_fixed)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_fixed_t.png'.format(output_dir, epoch, i), tmp)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), None, None, x_fixed_t, netGS, netGM, netGT,
                                                           netE)
                z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE)

                with torch.no_grad():
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(x_fixed_t[j:j+1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_pos_fixed.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_pos_s.png'.format(output_dir, epoch, i), netGS(z_g_k))
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_pos_m.png'.format(output_dir, epoch, i), netGM(z_g_k))
                    gen_text = netGT(z_g_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_pos_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_0_s.png'.format(output_dir, epoch, i), netGS(z_e_0))
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_0_m.png'.format(output_dir, epoch, i), netGM(z_e_0))
                    gen_text = netGT(z_e_0)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_0_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_k_s.png'.format(output_dir, epoch, i), netGS(z_e_k))
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_k_m.png'.format(output_dir, epoch, i), netGM(z_e_k))
                    gen_text = netGT(z_e_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_neg_k_t.png'.format(output_dir, epoch, i), tmp)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_fixed_s.png'.format(output_dir, epoch, i), netGS(z_fixed))
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_fixed_m.png'.format(output_dir, epoch, i), netGM(z_fixed))
                    gen_text = netGT(z_fixed)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_fixed_t.png'.format(output_dir, epoch, i), tmp)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), x_fixed_s, x_fixed_m, x_fixed_t, netGS,
                                                           netGM, netGT, netE)
                with torch.no_grad():
                    plot('{}/samples/svhn/{:>06d}_{:>06d}_x_z_pos_s_jon.png'.format(output_dir, epoch, i), netGS(z_g_k))
                    plot('{}/samples/mnist/{:>06d}_{:>06d}_x_z_pos_m_jon.png'.format(output_dir, epoch, i),
                         netGM(z_g_k))
                    gen_text = netGT(z_g_k)
                    for j in range(tmp.shape[0]):
                        tmp[j] = text_to_pil(gen_text[j:j + 1], img_size, alphabet)
                    plot('{}/samples/text/{:>06d}_{:>06d}_x_z_pos_t_jon.png'.format(output_dir, epoch, i), tmp)

        # Schedule
        lr_scheduleE.step(epoch=epoch)
        lr_scheduleGS.step(epoch=epoch)
        lr_scheduleGM.step(epoch=epoch)
        lr_scheduleGT.step(epoch=epoch)
        # Stats
        if epoch % args.n_stats == 0:
            stats['loss_g_s'].append(loss_g_s.item())
            stats['loss_g_m'].append(loss_g_m.item())
            stats['loss_g_t'].append(loss_g_t.item())
            stats['loss_e'].append(loss_e.item())
            stats['en_neg'].append(en_neg.data.item())
            stats['en_pos'].append(en_pos.data.item())
            stats['grad_norm_g_s'].append(grad_norm_g_s)
            stats['grad_norm_g_m'].append(grad_norm_g_m)
            stats['grad_norm_g_t'].append(grad_norm_g_t)
            stats['grad_norm_e'].append(grad_norm_e)
            stats['z_g_grad_norm_s'].append(z_g_grad_norm_s.item())
            stats['z_g_grad_norm_m'].append(z_g_grad_norm_m.item())
            stats['z_g_grad_norm_t'].append(z_g_grad_norm_t.item())
            stats['z_e_grad_norm'].append(z_e_grad_norm.item())
            stats['z_e_k_grad_norm'].append(z_e_k_grad_norm.item())
            stats['fid'].append(fid)
            interval.append(epoch + 1)
            plot_stats(output_dir, stats, interval)


        if epoch % args.n_metrics == 0 and epoch > 0:
            import fid_score
            s1 = []
            for _ in range(int(10000 / 100)):
                z_0 = sample_p_0().cuda()
                z_k = sample_langevin_prior_z(Variable(z_0), netE)[0]
                x_samples_s = netGS(z_k).clamp(min=0., max=1.)
                s1.append(x_samples_s)
            s1 = torch.cat(s1)
            fid = fid_score.compute_fid(x_train=None, x_samples=s1,
                                        path='/Tian-ds/hli136/project/latent-space-EBM-prior-main/fid_real/fid_stats_svhn_train.npz')
            print('Fid', fid)

            if fid < min_fid:
                min_fid = fid
            print('Min Fid', min_fid)


        acc_m = 0
        acc_s = 0
        correct = 0
        total = 0
        if epoch >= 0:
            for ind, x in enumerate(test_loader):
                total += xm.shape[0]
                xm, xs, xt, xl = x;
                xm = xm.cuda()
                xs = xs.cuda()
                xt = xt.cuda()
                xl = xl.cuda()

                # Cross
                # MNIST
                z_g_0 = sample_p_0(n=batch_size_fixed)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), None, xm, None, netGS, netGM, netGT,
                                                           netE)
                xs_p = netGS(z_g_k)
                ys = clf_svhn(xs_p)
                top_pred_s = ys.argmax(1, keepdim=True)
                correct_s = top_pred_s.eq(xl.view_as(top_pred_s)).sum()
                acc_s += correct_s.float()

                # SVHN
                z_g_0 = sample_p_0(n=batch_size_fixed)

                z_g_k, _, _, _, _ = sample_langevin_post_z(Variable(z_g_0), xs, None, None, netGS, netGM, netGT,
                                                           netE)
                xm_p = netGM(z_g_k)
                ym = clf_mnist(xm_p)
                top_pred_m = ym.argmax(1, keepdim=True)
                correct_m = top_pred_m.eq(xl.view_as(top_pred_m)).sum()
                acc_m += correct_m.float()


            cross_mnist = acc_m / total
            cross_svhn = acc_s / total
            if cross_mnist > max_cross_mnist:
                max_cross_mnist = cross_mnist
            if cross_svhn > max_cross_svhn:
                max_cross_svhn = cross_svhn
            print('Acc Mnist', cross_mnist)
            print('Acc Svhn', cross_svhn)

            # Joint
            for ind in range(100):
                z_e_0 = sample_p_0(n=batch_size_fixed)
                z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE)

                xm_p = netGM(z_e_k)
                xs_p = netGS(z_e_k)

                ym = clf_mnist(xm_p)
                ys = clf_svhn(xs_p)

                top_pred_m = ym.argmax(1, keepdim=True)
                top_pred_s = ys.argmax(1, keepdim=True)

                correct += top_pred_s.eq(top_pred_m.view_as(top_pred_s)).sum()

            joint_co = correct / (100 * batch_size_fixed)
            if joint_co > max_joint_co:
                max_joint_co = joint_co
            print('Joint', joint_co)


        # Plot

        # Ckpt
        if epoch > 0 and epoch % args.n_ckpt == 0:
            save_dict = {
                'epoch': epoch,
                'netE': netE.state_dict(),
                'optE': optE.state_dict(),
                'netGS': netGS.state_dict(),
                'netGM': netGM.state_dict(),
                'netGT': netGT.state_dict(),
                'optGS': optGS.state_dict(),
                'optGM': optGM.state_dict(),
                'optGT': optGT.state_dict(),
            }
            torch.save(save_dict, '{}/ckpt/ckpt_{:>06d}.pth'.format(output_dir, epoch))


    return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}



##########################################################################################################
## Metrics


##########################################################################################################
## Plots

import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    # f = plt.figure(figsize=(20, len(content) * 5))
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        axs[j].plot(interval, v)
        axs[j].set_ylabel(k)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)



##########################################################################################################
## Other

def get_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def print_gpus():
    os.system('nvidia-smi -q -d Memory > tmp')
    tmp = open('tmp', 'r').readlines()
    for l in tmp:
        print(l, end = '')


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    free_gpu = np.argmax(memory_available)
    print('set gpu', free_gpu, 'with', np.max(memory_available), 'mb')
    return free_gpu


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


##########################################################################################################
## Main

def makedirs_exp(output_dir):
    os.makedirs(output_dir + '/samples')
    os.makedirs(output_dir + '/samples/svhn')
    os.makedirs(output_dir + '/samples/mnist')
    os.makedirs(output_dir + '/samples/text')
    os.makedirs(output_dir + '/ckpt')

def main():

    print_gpus()

    fs_prefix = './' 

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = pygrid.get_output_dir(exp_id, fs_prefix=fs_prefix)

    # run
    copy_source(__file__, output_dir)
 #   opt = {'job_id': int(0), 'status': 'open', 'device': get_free_gpu()}
    train(output_dir, output_dir, {})


if __name__ == '__main__':
    main()
