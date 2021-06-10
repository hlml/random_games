import torchvision
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import tqdm
from torch.autograd import Variable
import argparse
import os
import gc
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
data_path = '/home/hattie/scratch/datasets/'
use_cuda = torch.cuda.is_available()

from models import ConvBlock, Flatten, LeNet, MnistConvNet, RandomGame, RandomGamePos, WideResNet

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_normal', action='store_true', help='train baseline model without game')  
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--severity', type=int, default=0, help='0 for all severity')
    parser.add_argument('--init_type', type=str, default = 'prev', choices=('prev','true','reinit'),
                        help='init type for rand training')  
    parser.add_argument('--use_augmix', action='store_true', help='train with augmix data augmentation')
    parser.add_argument(
        '--no_jsd',
        '-nj',
        action='store_true',
        help='Turn off JSD consistency loss.')
    return parser


def test(loader, model, save=False, bn_eval=True):
    
    if bn_eval: # forward prop data twice to update BN running averages
        model.train()
        for _ in range(2):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                _ = (model(inputs, True))

    model.eval()
    correct, total, total_loss = 0,0,0
    tot_iters = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = (model(inputs, True))

            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += torch.sum(predicted.eq(targets.data)).cpu()
            total_loss += model.loss_fn(outputs, targets).item()

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    loss = total_loss/tot_iters
    return loss, acc



class CIFAR10CDataset(Dataset):
    
    """CIFAR10-C dataset."""

    def __init__(self, root, corruption, severity=None, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if severity == None:
            self.data = np.load(os.path.join(root, corruption + '.npy'))
            self.targets = torch.tensor(np.load(os.path.join(root, 'labels.npy')))
        
        else:
            self.data = np.load(os.path.join(root, corruption, 'severity' + str(severity), 'images.npy'))
            self.targets = torch.tensor(np.load(os.path.join(root, corruption, 'severity' + str(severity), 'labels.npy')))
            self.severity = severity
            
        self.root = root
        self.corruption = corruption
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def main():    
    parser = make_parser()
    args = parser.parse_args()    

    model = RandomGame(model_func=WideResNet, num_class=10, depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    if use_cuda:
        model.cuda()

    if args.use_augmix:
        trans=transforms.Compose(
                  [transforms.ToTensor(),
                   transforms.Normalize([0.5] * 3, [0.5] * 3)])
    else:
        trans=transforms.Compose(
                        [transforms.ToTensor()])
#         trans = transforms.Compose(
#                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])

    model_name = 'cifar10_convnet'
    if args.train_normal:
        model_name += '_NORMAL'
    model_name += '_trainfirst_iterT1_iterR1_iterRsb1_gam'
    if args.train_normal:
        model_name += '0_alf1'
    else:
        model_name += '0.003_alf1.0'
    model_name += '_lrT0.1_lrR0.1_momT0.9_momR0_l2T0.0005_l2R'
    if args.train_normal:
        model_name += '0_ep100_seed'
    else:
        model_name += '0.0_dgm0.95per300_ep100_simt_seed'
        
    if args.use_augmix:
        if args.no_jsd:
            model_name2 = '_AUGMIX_mw3_md-1_sev3_fromfresh_clipg'
        else:
            model_name2 = '_AUGMIX_mw3_md-1_sev3_jsd_fromfresh_clipg'            
    else:
        model_name2 = '_fromfresh_clipg'


    if args.epoch == 100:
        args.epoch = 'final_model'

    results_acc = np.zeros((5,19))
    results_loss = np.zeros((5,19))

    for i, cpt in enumerate(['brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'pixelate', 'shot_noise', 'spatter', 'zoom_blur', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'saturate', 'snow', 'speckle_noise']):    
        if args.severity == 0:
            trainset = CIFAR10CDataset(root=data_path + 'CIFAR-10-C/', corruption=cpt, severity=None, transform=trans)
        else:
            trainset = CIFAR10CDataset(root=data_path + 'CIFAR-10-C/processed_data/', corruption=cpt, severity=args.severity, transform=trans)            
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=5)

        for seed in range(5):

            model_checkpoint = torch.load('/home/hattie/random_games/clean_exp_cifar_fast_clip/' + model_name + str(seed) + '_inittype' + args.init_type + model_name2 + '/' + str(args.epoch) + '.tar')

            model.load_state_dict(model_checkpoint['state_dict'])
            loss, acc = test(trainloader, model, save=False, bn_eval=False)
            results_acc[seed, i] = acc
            results_loss[seed, i] = loss


    #save metrics
    save_path = '/home/hattie/random_games/clean_exp_cifar_fast_clip_PERF/' + model_name + str(0) + '_inittype' + args.init_type + model_name2 + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('saving to workdir %s' % save_path)

    with h5py.File(os.path.join(save_path, 'sev' + str(args.severity) +'_ep' + str(args.epoch) + '_perf_metrics.h5'), 'w') as f:

        f.attrs['keys'] = ['results_acc', 'results_loss']

        loss_dset = f.create_dataset('results_loss', results_loss.shape)
        loss_dset[:] = results_loss        
        acc_dset = f.create_dataset('results_acc', results_acc.shape)
        acc_dset[:] = results_acc

    
if __name__ == '__main__':
    main()