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
    parser.add_argument('--init_type', type=str, default = 'prev', choices=('prev','true','reinit'),
                        help='init type for rand training')  
    parser.add_argument('--use_augmix', action='store_true', help='train with augmix data augmentation')
    parser.add_argument(
        '--no_jsd',
        '-nj',
        action='store_true',
        help='Turn off JSD consistency loss.')
    return parser


def test(loader, model, bn_eval=True):
    
    if bn_eval: # forward prop data twice to update BN running averages
        model.train()
        for _ in range(2):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                _ = (model(inputs, False))

    model.eval()
    correct, total, total_loss = 0,0,0
    tot_iters = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = (model(inputs, False))

            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += torch.sum(predicted.eq(targets.data)).cpu()
            total_loss += model.loss_fn(outputs, targets).item()

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)
    loss = total_loss/tot_iters
    return loss, acc


    
def train_rand_loop(model, optimizer_R, inputs, rand_labels, scheduler_R=None, pos_weight=False,
                    reinit=False, init_type='prev', num_iter_rand_sb=1, mse_loss_func=None, no_jsd=True):
    
    if reinit:
        init_rand_weights(model, init_type=init_type)
        
    if not no_jsd:
        inputs = torch.split(inputs, rand_labels.size(0))[0]
        
    model.train_rand()
    for k in range(num_iter_rand_sb):
        optimizer_R.zero_grad()
        outputs_rand = model(inputs, False)
        if mse_loss_func is not None:
            rand_loss=mse_loss_func(outputs_rand, rand_labels)
        else:
            rand_loss=model.loss_fn(outputs_rand, rand_labels)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        rand_loss.backward()
        optimizer_R.step()
        if pos_weight:
            for p in model.rand_classifier.parameters():
                p.data.clamp_(0)
                
    if scheduler_R is not None:
        scheduler_R.step()
#     return rand_loss

def train(model, trainloader, optimizer_R):
    loss_tracker = []
    acc_tracker = []
    for epoch in range(100):  # loop over the dataset multiple times

        correct=0
        total=0
        correct_rand=0
        running_loss_true = 0.0
        running_loss_rand = 0.0
        for i, data in enumerate(trainloader):
            model.train()
            inputs, rand_labels = data

            if use_cuda:
                inputs, rand_labels = inputs.cuda(), rand_labels.cuda()
            train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=False,
                    reinit=False, init_type='prev', scheduler_R=None,
                    num_iter_rand_sb=1, mse_loss_func=None)

        with torch.no_grad():
            outputs_rand = model(inputs, False)
            rand_loss=model.loss_fn(outputs_rand, rand_labels)

            total += rand_labels.size(0)
            _, predicted_rand = torch.max(nn.Softmax(dim=1)(outputs_rand).data, 1)
            correct_rand += predicted_rand.eq(rand_labels.data).cpu().sum()


        loss = rand_loss.item()
        acc = correct_rand.numpy() / total * 100
        loss_tracker.append(loss)
        acc_tracker.append(acc)
    
    return loss_tracker, acc_tracker


def main():    
    parser = make_parser()
    args = parser.parse_args()    
    
    #set random seeds
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    model = RandomGame(model_func=WideResNet, num_class=10, depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    if use_cuda:
        model.cuda()
        
    orig_weights=[]
    for name, param in model.named_parameters():
        if 'rand' in name:
            orig_weights.append(param.data.clone())


    if args.use_augmix:
#         trans=transforms.Compose(
#                   [transforms.ToTensor(),
#                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
        trans = transforms.Compose(
          [transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32, padding=4)])
    else:
        trans=transforms.Compose(
                        [transforms.ToTensor()])


    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=trans)
    batch_split, rest_split = 500, 9500
    test_set, rest_set = torch.utils.data.random_split(test_set, [batch_split, rest_split])
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_split, shuffle=False, num_workers=4, pin_memory=True)

    random_labels = torch.randint(0, 10, (batch_split,))
    random_idx = np.random.choice(testloader.dataset.indices, batch_split, replace=False)
    
    temp_targets = torch.tensor(testloader.dataset.dataset.targets)
    temp_targets[random_idx] = random_labels
    testloader.dataset.dataset.targets=list(temp_targets.numpy())

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
            model_name2 = '_AUGMIX_mw3_md-1_sev3_fromfresh'
        else:
            model_name2 = '_AUGMIX_mw3_md-1_sev3_jsd_fromfresh'            
    else:
        model_name2 = '_fromfresh'

    results_acc = np.zeros((5,11,100))
    results_loss = np.zeros((5,11,100))

    for seed in range(5):
        for i, ep in enumerate(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', 'final_model']):
            model_checkpoint = torch.load('/home/hattie/random_games/clean_exp_cifar_fast/' + model_name + str(seed) + '_inittype' + args.init_type + model_name2 + '/' + ep + '.tar')

            model.load_state_dict(model_checkpoint['state_dict'])
            
#             for name, layer in model.named_children():
#                 if 'rand' in name:
#                     layer[0].reset_parameters()


            t=0
            for name, param in model.named_parameters():
                if 'rand' in name:
                    param.data=orig_weights[t]
                    t+=1

            optimizer_R = torch.optim.SGD(model.rand_classifier.parameters(), lr=0.1)
            loss_tracker, acc_tracker = train(model, testloader, optimizer_R)
            
            results_acc[seed, i, :] = np.array(acc_tracker)
            results_loss[seed, i, :] = np.array(loss_tracker)


    #save metrics
    save_path = '/home/hattie/random_games/clean_exp_cifar_fast_RAND_PERF/' + model_name + str(0) + '_inittype' + args.init_type + model_name2 + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('saving to workdir %s' % save_path)

    with h5py.File(os.path.join(save_path, 'ALL_same_rand_LR1_TRANS_perf_metrics.h5'), 'w') as f:

        f.attrs['keys'] = ['results_acc', 'results_loss']

        loss_dset = f.create_dataset('results_loss', results_loss.shape)
        loss_dset[:] = results_loss        
        acc_dset = f.create_dataset('results_acc', results_acc.shape)
        acc_dset[:] = results_acc

    
if __name__ == '__main__':
    main()