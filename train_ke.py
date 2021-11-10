import torchvision
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import argparse
import os
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
import wandb

from models import LeNet, MnistConvNet, RandomGame, WideResNet
from dataset import MNISTCDataset
import augmentations

data_path = '/home/hattie/scratch/datasets/'
use_cuda = torch.cuda.is_available()

       
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
        
    parser.add_argument('--train_data', type=str, default = 'mnist', choices=('mnist','cifar10'),
                        help='name of training dataset')
    parser.add_argument('--model_choice', type=str, default = 'lenet')
    parser.add_argument('--num_classes', type=int, default=10)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    
    parser.add_argument('--train_label_noise', type=float, default = 0, help='perc of label noise in train data')
    parser.add_argument('--train_mislabeled', type=float, default = 0, help='perc of mislabeled data points in train data')
    parser.add_argument('--clip_grad', action='store_true', help='clip gradients')  
    
    parser.add_argument('--init_type', type=str, default = 'prev', choices=('prev','true','reinit'),
                        help='init type for rand training')   
    
    parser.add_argument('--mrl', type=str, default = None, help = 'comma-separated list for layers in rand classifier (if more than one)')
    
    parser.add_argument('--lr', type=float, default=.1, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')        
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--l2', type=float, default=0, help='weight decay rate')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    
    
    parser.add_argument('--model_path', type=str, default=None, help='load a pretrained model from this path')
    parser.add_argument('--save_path', type=str, default='exp_logged', help='save model object to this path')
    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--save_every', type=int, default=1, help='save model every n epochs')
    parser.add_argument('--save_init_model', action='store_true', help='save initialization model state')
    
    parser.add_argument('--model_name', type=str, default='exp')
    parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
    
    #augmix options
    parser.add_argument('--use_augmix', action='store_true', help='train with augmix data augmentation')
    parser.add_argument(
        '--mixture_width',
        default=3,
        type=int,
        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument(
        '--mixture_depth',
        default=-1,
        type=int,
        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument(
        '--aug_severity',
        default=3,
        type=int,
        help='Severity of base augmentation operators')
    parser.add_argument(
        '--no_jsd',
        '-nj',
        action='store_true',
        help='Turn off JSD consistency loss.')
    parser.add_argument(
        '--all_ops',
        '-all',
        action='store_true',
        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    return parser



def test(loader, model, save=False, bn_eval=True):
    
    if bn_eval: # forward prop data twice to update BN running averages
        model.train()
        for _ in range(2):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                _ = (model(inputs))

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


def get_all_loss(model, outputs, labels, no_jsd=True):
    
    if no_jsd:
        true_loss=model.loss_fn(outputs, labels)
    else:
        logits_clean, logits_aug1, logits_aug2 = torch.split(outputs, int(outputs.size(0)/3))
        
        # Cross-entropy is only computed on clean images
        loss = F.cross_entropy(logits_clean, labels)

        p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    
    return loss


def init_rand_weights(model, init_type='prev'):
    if init_type == 'true':
        cur_weights=[]
        for name, param in model.named_parameters():
            if 'true' in name:
                cur_weights.append(param.data.clone())
                
        t=0
        for name, param in model.named_parameters():
            if 'rand' in name:
                param.data=cur_weights[t]
                t+=1
    elif init_type == 'reinit':
        for name, layer in model.named_children():
            if 'rand' in name:
                layer[0].reset_parameters()

def get_model_state(model, epoch, optimizer, scheduler=None):
    state = {
        'epoch':           epoch,
        'state_dict':      model.state_dict(),
        'optimizer':       optimizer.state_dict(),
        'scheduluer':      scheduler.state_dict()
    }
    return state

    
def train(args, model, optimizer, trainloader, validloader, workdir, scheduler=None):
    
    iter_counter=0    
    
    #prepare metric trackers
    validloss_all = []
    validacc_all = []
    trainloss_all = []
    trainacc_all = []
    
    correct=0
    total=0
        
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
    #     for i, data in enumerate(trainloader):
            model.train()
            inputs, labels = data
            
            if not args.no_jsd and args.use_augmix:
                inputs = torch.cat(inputs, 0)

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()        

            outputs = model(inputs)
            
            if args.no_jsd:
                true_loss=model.loss_fn(outputs, labels)
            elif args.use_augmix:
                logits_clean, logits_aug1, logits_aug2 = torch.split(outputs, int(outputs.size(0)/3))

                # Cross-entropy is only computed on clean images
                loss = F.cross_entropy(logits_clean, labels)

                p_clean, p_aug1, p_aug2 = F.softmax(
                  logits_clean, dim=1), F.softmax(
                      logits_aug1, dim=1), F.softmax(
                          logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            else:
                loss=model.loss_fn(outputs, labels)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            #evaluate on true labels
            with torch.no_grad():
                if not args.no_jsd and args.use_augmix:
                    inputs = torch.split(inputs, labels.size(0))[0]
                outputs = model(inputs)

                _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()


                # print statistics
                running_loss += loss.item()

                if i % args.print_every == args.print_every -1:   
                    validloss, validacc = test(validloader, model, bn_eval=False)

                    validloss_all.append(validloss)
                    validacc_all.append(validacc)
                    trainloss_all.append(running_loss / args.print_every)
                    trainacc_all.append(correct.numpy() / total * 100)
                    print('[%d, %5d] true loss: %.3f, train acc: %.3f, val loss: %.3f, val acc: %.3f' %
                          (epoch + 1, i + 1, 
                           running_loss / args.print_every, 
                           correct.numpy() / total * 100, validloss, validacc))


                    wandb.log({'Epoch': epoch + 1, 'Iter': i+1, 'Train Loss': running_loss / args.print_every, 'Train Accuracy': correct.numpy() / total * 100, 'Valid Loss': validloss, 'Valid Accuracy':validacc})

                    running_loss = 0.0
                    correct = 0
                    total = 0


        if epoch % args.save_every==0:
            outfile = os.path.join(workdir, '{:d}.tar'.format(epoch))
            torch.save(get_model_state(model, epoch, optimizer, scheduler), outfile)
    
    #save final model
    outfile = os.path.join(workdir, 'final_model.tar')
    torch.save(get_model_state(model, epoch, optimizer, scheduler), outfile)

    return trainloss_all, trainacc_all, validloss_all, validacc_all#, testloss_all, testacc_all
    
    
        
def main():    
    parser = make_parser()
    args = parser.parse_args()

    def aug(image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.

        Returns:
        mixed: Augmented and mixed image.
        """
        aug_list = augmentations.augmentations
        if True:#args.all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * args.mixture_width)) #3-args.mixture_width
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(args.mixture_width): #3-args.mixture_width
            image_aug = image.copy()
            depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, args.aug_severity) #3-args.aug_severity
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed


    class AugMixDataset(torch.utils.data.Dataset):
        """Dataset wrapper to perform AugMix augmentation."""

        def __init__(self, dataset, preprocess, no_jsd=False):
            self.dataset = dataset
            self.preprocess = preprocess
            self.no_jsd = no_jsd

        def __getitem__(self, i):
            x, y = self.dataset[i]
            if self.no_jsd:
                return aug(x, self.preprocess), y
            else:
                im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                          aug(x, self.preprocess))
            return im_tuple, y

        def __len__(self):
            return len(self.dataset)
        
    class NonAugMixDataset(torch.utils.data.Dataset):
        """Dataset wrapper to perform AugMix augmentation."""

        def __init__(self, dataset, preprocess):
            self.dataset = dataset
            self.preprocess = preprocess

        def __getitem__(self, i):
            x, y = self.dataset[i]
            return self.preprocess(x), y

        def __len__(self):
            return len(self.dataset)
    
    
    
    #set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     random.seed(args.seed)
    np.random.seed(args.seed)
    
    if not args.use_augmix:
        args.no_jsd = True
    
    #model name creation
    model_run_name=args.train_data
#     if args.test_data is not None:
#         model_run_name = model_run_name + '_' + args.test_data
#     else:
#         model_run_name = model_run_name + '_origtest'
    model_run_name = model_run_name + '_' + args.model_choice
    model_run_name = model_run_name + '_lr' + str(args.lr)
    model_run_name = model_run_name + '_mom' + str(args.mom)
    model_run_name = model_run_name + '_l2' + str(args.l2)
    model_run_name = model_run_name + '_ep' + str(args.num_epochs)
    model_run_name = model_run_name + '_seed' + str(args.seed)
    model_run_name = model_run_name + '_inittype' + args.init_type
    if args.use_augmix:
        model_run_name = model_run_name + '_AUGMIX'
        model_run_name = model_run_name + '_mw' + str(args.mixture_width) + '_md' + str(args.mixture_depth) + '_sev' + str(args.aug_severity)
    if args.all_ops:
        model_run_name = model_run_name + '_allops'
    if not args.no_jsd:
        model_run_name = model_run_name + '_jsd'
    if args.clip_grad:
        model_run_name = model_run_name + '_clipg'
        
        
    if len(args.group_vars) > 0:
        if len(args.group_vars) == 1:
            group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
        else:
            group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
            for var in args.group_vars[1:]:
                group_name = group_name + '_' + var + str(getattr(args, var))
        wandb.init(project="cifar_forget",
               group=args.model_name,
               name=group_name)
        for var in args.group_vars:
            wandb.config.update({var:getattr(args, var)})
            
    workdir = os.path.join(args.save_path, model_run_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    print('saving to workdir %s' % workdir)
    
    #load data
    if args.train_data == 'mnist':
        trans = ([transforms.ToTensor()])
        train_transform = transforms.Compose(trans)
        preprocess = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize([0.5], [0.5])])
    if args.train_data == 'cifar10':
        if args.use_augmix:
            # Load datasets
            train_transform = transforms.Compose(
              [transforms.RandomHorizontalFlip(),
               transforms.RandomCrop(32, padding=4)])
            preprocess = transforms.Compose(
              [transforms.ToTensor(),
               transforms.Normalize([0.5] * 3, [0.5] * 3)])
            test_transform = preprocess
        else:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
            test_transform = transforms.Compose(
                [transforms.ToTensor()])
    
    if args.train_data == 'mnist':
        fulltrainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=train_transform)
    if args.train_data == 'cifar10':
        fulltrainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        
    train_split, val_split = int(len(fulltrainset)*0.9), int(len(fulltrainset)*0.1)
    train_set, valset = torch.utils.data.random_split(fulltrainset, [train_split, val_split])
    
    if args.use_augmix:
        train_set = AugMixDataset(train_set, preprocess, no_jsd=args.no_jsd)
        valset = NonAugMixDataset(valset, preprocess)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                              num_workers=10, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=10, pin_memory=True)
   
    
#     if args.test_data is not None:
#         if args.train_data == 'mnist':
#             test_set = MNISTCDataset(root=os.path.join(data_path, args.test_data), train=False, transform=train_transform)
#         testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     else:
#         if args.train_data == 'mnist':
#             test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=train_transform)
#         elif args.train_data == 'cifar10':
#             test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
#         testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)    
    
    
    #initialize models
    if args.model_choice == 'lenet':
#         model = RandomGame(model_func=LeNet, num_class = args.num_classes, multi_rand_layer=multi_rand_layer)
        raise NotImplementedError
            
    elif args.model_choice == 'convnet':
        if args.train_data == 'mnist':
#             model = RandomGame(model_func=MnistConvNet, num_class = args.num_classes, multi_rand_layer=multi_rand_layer)
            raise NotImplementedError
        if args.train_data == 'cifar10':
            model = RWideResNet(depth=40, num_classes=args.num_classes, widen_factor=2, dropRate=0.0)

            
    # Optimizers
    if args.mom > 0:
        nest_flag = True
    else:
        nest_flag = False
    optimizer = torch.optim.SGD(model, lr = args.lr, momentum=args.mom, weight_decay=args.l2, nesterov=nest_flag)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))


    if args.train_data == 'mnist':
        scheduler=None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.num_epochs * len(trainloader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr))
            
    if args.model_path is not None:
        model_checkpoint = torch.load(args.model_path)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        if args.train_data != 'mnist':
            scheduler.load_state_dict(model_checkpoint['scheduler'])

    if args.save_init_model and args.model_path is None:
        outfile = os.path.join(workdir, 'init_model.tar')
        torch.save({'epoch':-1, 'state_dict':model.state_dict()}, outfile)
                              
    if use_cuda:
        model.cuda()

    trainloss_all, trainacc_all, validloss_all, validacc_all = train(args, model, 
            optimizer, trainloader, validloader, 
            workdir, 
            scheduler=scheduler)

    #save metrics
    with h5py.File(os.path.join(workdir, 'perf_metrics.h5'), 'w') as f:
        f.attrs['print_every'] = args.print_every
        f.attrs['num_epochs'] = args.num_epochs
        f.attrs['keys'] = ['train_loss', 'train_acc', 'val_loss', 'val_acc']#, 'test_loss', 'test_acc']

        tl_dset = f.create_dataset('train_loss', np.array(trainloss_all).shape)
        tl_dset[:] = np.array(trainloss_all)
        ta_dset = f.create_dataset('train_acc', np.array(trainacc_all).shape)
        ta_dset[:] = np.array(trainacc_all)
        vl_dset = f.create_dataset('val_loss', np.array(validloss_all).shape)
        vl_dset[:] = np.array(validloss_all)
        va_dset = f.create_dataset('val_acc', np.array(validacc_all).shape)
        va_dset[:] = np.array(validacc_all)
#         el_dset = f.create_dataset('test_loss', np.array(testloss_all).shape)
#         el_dset[:] = np.array(testloss_all)
#         ea_dset = f.create_dataset('test_acc', np.array(testacc_all).shape)
#         ea_dset[:] = np.array(testacc_all)

    wandb.run.finish()
                              
if __name__ == '__main__':
    main()