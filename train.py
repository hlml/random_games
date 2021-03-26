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

from models import LeNet, MnistConvNet, RandomGame, RandomGamePos
from dataset import MNISTCDataset

data_path = '../datasets/'
use_cuda = torch.cuda.is_available()

       
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
        
    parser.add_argument('--train_data', type=str, default = 'mnist', choices=('mnist','cifar10'),
                        help='name of training dataset')
    parser.add_argument('--test_data', type=str, default = None, help='path to OOD dataset')
    parser.add_argument('--model_choice', type=str, default = 'lenet')
    parser.add_argument('--train_true', type=int, default=1, choices=(1,0), help='if 1, start by training true model, else start by training rand model')
    parser.add_argument('--mse', type=int, default=0, choices=(1,0), help='use mse loss for rand model')
    
    parser.add_argument('--num_iter_true', type=int, default=1, help='number of iters to train true labels in one round')
    parser.add_argument('--num_iter_rand', type=int, default=1, help='number of iters to train random labels in one round')
    parser.add_argument('--num_iter_rand_sb', type=int, default=1, help='number of iters to train random labels on one batch')
    parser.add_argument('--num_classes', type=int, default=10)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--gamma', type=float, required=True, help='weight of random loss')
    parser.add_argument('--alpha', type=float, required=True, help='weight of true loss')
    
    parser.add_argument('--train_label_noise', type=float, default = 0, help='perc of label noise in train data')
    parser.add_argument('--decay_gamma_every', type=int, default=100, help='decay gamma every n iterations')
    parser.add_argument('--decay_gamma_rate', type=float, default=1, help='decay gamma by')
    parser.add_argument('--simultaneous', action='store_true', help='simultaneously train random first and true second on the same batch')   
    parser.add_argument('--train_separately', action='store_true', help='train encoder and classifiers separately for true')   
    parser.add_argument('--pos_rand', action='store_true', help='use positive weights for random classifier')    
    parser.add_argument('--init_type', type=str, default = 'prev', choices=('prev','true','reinit'),
                        help='init type for rand training')   
    parser.add_argument('--random_per_batch', action='store_true', help='generate new random labels for every batch')
    parser.add_argument('--consistent_rand', action='store_true', help='generate consistent random labels for every batch')
    parser.add_argument('--train_rand_true', action='store_true', help='train by predicting true labels with rand model')
    
    parser.add_argument('--mrl', type=str, default = None, help = 'comma-separated list for layers in rand classifier (if more than one)')
    
    parser.add_argument('--lr_true', type=float, default=.1, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')        
    parser.add_argument('--lr_rand', type=float, default=.1, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')    
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--l2_true', type=float, default=0, help='weight decay rate')
    parser.add_argument('--l2_rand', type=float, default=0, help='weight decay rate')
    
    parser.add_argument('--model_path', type=str, default=None, help='load a pretrained model from this path')
    parser.add_argument('--save_path', type=str, default='exp_logged', help='save model object to this path')
    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--save_every', type=int, default=1, help='save model every n epochs')
    parser.add_argument('--save_init_model', action='store_true', help='save initialization model state')
    
    parser.add_argument('--model_name', type=str, default='exp')
    parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")

    #unused params
    parser.add_argument('--decay_lr', action='store_true', help='decay learning rate')
    parser.add_argument('--decay_schedule', type=str, default = '10,20,50,-1', help = 'comma-separated list')
    parser.add_argument('--mom', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--no_shuffle', action='store_true')

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
#     if save and acc > best_acc:
#         best_acc = acc
#         print('Saving best model..')
#         state = {
#             'model': model0,
#             'epoch': epoch
#         }
#         with open(args.save_dir + '/best_model.pt', 'wb') as f:
#             torch.save(state, f)
    return loss, acc

def get_all_loss(model, outputs_true, outputs_rand, labels, rand_labels, alpha, gamma, mse_loss_func=None):
    
    true_loss=model.loss_fn(outputs_true, labels)
    if mse_loss_func is not None:
        rand_loss=mse_loss_func(outputs_rand, rand_labels)
    else:
        rand_loss=model.loss_fn(outputs_rand, rand_labels)
    loss = alpha*true_loss - gamma*rand_loss
    
    return true_loss, rand_loss, loss


def train_true_loop(model, optimizer_T, inputs, labels, rand_labels, alpha, gamma, mse_loss_func=None):
    # zero the parameter gradients
    optimizer_T.zero_grad()        
    model.train_true()
    outputs_true = model(inputs, True)
    outputs_rand = model(inputs, False)
    
    true_loss, rand_loss, loss = get_all_loss(model, outputs_true, outputs_rand, labels, rand_labels, 
                                              alpha, gamma, mse_loss_func)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    loss.backward()
    optimizer_T.step()
#     return true_loss, rand_loss, loss
    
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
        if isinstance(model, RandomGamePos):
            for name, layer in model.named_children():
                if 'rand' in name:
                    layer.reset_parameters()
        else:
            for name, layer in model.named_children():
                if 'rand' in name:
                    layer[0].reset_parameters()
                    
                    
def train_rand_loop(model, optimizer, inputs, rand_labels, pos_weight=False,
                    reinit=False, init_type='prev', num_iter_rand_sb=1, mse_loss_func=None):
    
    if reinit:
        init_rand_weights(model, init_type=init_type)
        
    model.train_rand()
    for k in range(num_iter_rand_sb):
        optimizer.zero_grad()
        outputs_rand = model(inputs, False)
        if mse_loss_func is not None:
            rand_loss=mse_loss_func(outputs_rand, rand_labels)
        else:
            rand_loss=model.loss_fn(outputs_rand, rand_labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        rand_loss.backward()
        optimizer.step()
        if pos_weight:
            for p in model.rand_classifier.parameters():
                p.data.clamp_(0)
#     return rand_loss

def train_true_classifier_loop(model, optimizer, inputs, labels, 
                    init_type='prev', num_iter_rand_sb=1):
    
    if init_type == 'reinit':
        for name, layer in model.named_children():
            if 'true' in name:
                layer.reset_parameters()
        
    model.train_true_classifier()
    for k in range(num_iter_rand_sb):
        optimizer.zero_grad()
        outputs_true = model(inputs, True)
        loss=model.loss_fn(outputs_true, labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        loss.backward()
        optimizer.step()
        
def train_encoder_loop(model, optimizer, inputs, labels, rand_labels, alpha, gamma, mse_loss_func=None):
    # zero the parameter gradients
    model.train_encoder()
    optimizer.zero_grad()        
    
    outputs_true = model(inputs, True)
    outputs_rand = model(inputs, False)
    
    true_loss, rand_loss, loss = get_all_loss(model, outputs_true, outputs_rand, labels, rand_labels, 
                                              alpha, gamma, mse_loss_func)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    loss.backward()
    optimizer.step()
#     return true_loss, rand_loss, loss


def train(args, model, optimizer_T, optimizer_R, 
          trainloader, validloader, testloader, rand_labels_train, 
          workdir, mse_loss_func=None, optimizer_TC=None, optimizer_E=None):
    train_true = args.train_true
    iter_counter=0    
    
    #prepare metric trackers
    validloss_all = []
    validacc_all = []
    testloss_all = []
    testacc_all = []
    trainloss_all = []
    trainacc_all = []
    
    correct=0
    total=0
    correct_rand=0
    
    if args.consistent_rand:
        new_mapping = dict(zip(range(args.num_classes), 
                           np.random.choice(args.num_classes, args.num_classes, replace=False)))
    
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_loss_true = 0.0
        running_loss_rand = 0.0
        for i, (data, rand_labels) in enumerate(zip(trainloader, rand_labels_train), 0):
    #     for i, data in enumerate(trainloader):
            model.train()
            inputs, labels = data
            
            if not args.random_per_batch and args.consistent_rand:
                new_rand_labels = [new_mapping[x] for x in labels.numpy()]
                rand_labels = torch.tensor(new_rand_labels)
            
            if args.random_per_batch:
                if args.consistent_rand:
                    new_mapping = dict(zip(range(args.num_classes), 
                                           np.random.choice(args.num_classes, args.num_classes, replace=False)))
                    new_rand_labels = [new_mapping[x] for x in labels.numpy()]
                    rand_labels = torch.tensor(new_rand_labels)
                else:
                    rand_labels = torch.randint(0, args.num_classes, labels.shape)
                
            if mse_loss_func is not None:
                rand_labels_oh = torch.nn.functional.one_hot(rand_labels).float()
                if use_cuda:
                    rand_labels_oh=rand_labels_oh.cuda()
    #         rand_labels = torch.randint(0, 10, labels.shape)
            if use_cuda:
                inputs, labels, rand_labels = inputs.cuda(), labels.cuda(), rand_labels.cuda()

            #train for one iteration
            
            #train simultaneously means training classifiers and then encoder (rand then true) on the same batch of data. this gives the classifiers a disadvantage because they go first, but maybe allow encoder to overfit more easily to the batch.
            if args.simultaneous:
                #if training separately, we train the classifiers first on the same batch of data, then train the encoder to maximize rand loss and minimize true loss
                #not supporting mse for this type for now
                if args.train_separately:
                    if args.train_true:
                        train_true_classifier_loop(model, optimizer_TC, inputs, labels,
                                                   init_type='prev', num_iter_rand_sb=args.num_iter_rand_sb) #train one loop only for true
                        train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=args.pos_rand,
                                        reinit=True, init_type=args.init_type,
                                        num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)
                    else:
                        train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=args.pos_rand,
                                        reinit=True, init_type=args.init_type,
                                        num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)
                        train_true_classifier_loop(model, optimizer_TC, inputs, labels,
                                                   init_type='prev', num_iter_rand_sb=args.num_iter_rand_sb) #train one loop only for true
                    if args.train_rand_true:
                        train_encoder_loop(model, optimizer_E, inputs, labels, labels, 
                                           args.alpha, -args.gamma, mse_loss_func=mse_loss_func)                        
                    else:
                        train_encoder_loop(model, optimizer_E, inputs, labels, rand_labels, 
                                           args.alpha, args.gamma, mse_loss_func=mse_loss_func)
                    
                #if training together, first update random classifier, then update the true classifier and encoder in one step
                else:
                    if mse_loss_func is not None:
                        if not train_true:
                            train_rand_loop(model, optimizer_R, inputs, rand_labels_oh, pos_weight=args.pos_rand,
                                            reinit=True, init_type=args.init_type,
                                            num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)
                        train_true_loop(model, optimizer_T, inputs, labels, rand_labels_oh,
                                        args.alpha, args.gamma, mse_loss_func=mse_loss_func)
                    else:
                        if not train_true:
                            train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=args.pos_rand,
                                            reinit=True, init_type=args.init_type,
                                            num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)
                        train_true_loop(model, optimizer_T, inputs, labels, rand_labels,
                                        args.alpha, args.gamma, mse_loss_func=mse_loss_func)
                
            else:
                if train_true:
                    #if training separately, treat train_true as training the encoder
                    if args.train_separately:
                        train_encoder_loop(model, optimizer_E, inputs, labels, rand_labels, 
                                           args.alpha, args.gamma, mse_loss_func=mse_loss_func)
                    else:
                        if mse_loss_func is not None:
                            train_true_loop(model, optimizer_T, inputs, labels, rand_labels_oh,
                                            args.alpha, args.gamma, mse_loss_func=mse_loss_func)
                        else:
                            train_true_loop(model, optimizer_T, inputs, labels, rand_labels,
                                            args.alpha, args.gamma, mse_loss_func=mse_loss_func)


                if not train_true:
                    #for train_separately: not supporting reinit here for now, not supporting mse
                    if args.train_separately:
                        train_true_classifier_loop(model, optimizer_TC, inputs, labels,
                                                   init_type=args.init_type, num_iter_rand_sb=1) #train one loop only for true
                        train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=args.pos_rand,
                                        reinit=True, init_type=args.init_type,
                                        num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)

                    else:
                        if iter_counter == 0:
                            reinit_flag = True
                        else:
                            reinit_flag = False
                        if mse_loss_func is not None:
                            train_rand_loop(model, optimizer_R, inputs, rand_labels_oh, pos_weight=args.pos_rand,
                                            reinit=reinit_flag, init_type=args.init_type,
                                            num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)
                        else:
                            train_rand_loop(model, optimizer_R, inputs, rand_labels, pos_weight=args.pos_rand,
                                            reinit=reinit_flag, init_type=args.init_type,
                                            num_iter_rand_sb=args.num_iter_rand_sb, mse_loss_func=mse_loss_func)

            #evaluate on true labels
            outputs_true = model(inputs, True)
            outputs_rand = model(inputs, False)
            
            if mse_loss_func is not None:
                true_loss, rand_loss, loss = get_all_loss(model, outputs_true, outputs_rand, labels, rand_labels_oh, 
                                                  args.alpha, args.gamma, mse_loss_func)
            else:
                if args.train_rand_true:
                    true_loss, rand_loss, loss = get_all_loss(model, outputs_true, outputs_rand, labels, labels, 
                                                      args.alpha, -args.gamma, mse_loss_func)
                else:
                    true_loss, rand_loss, loss = get_all_loss(model, outputs_true, outputs_rand, labels, rand_labels, 
                                                      args.alpha, args.gamma, mse_loss_func)

            _, predicted = torch.max(nn.Softmax(dim=1)(outputs_true).data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            _, predicted_rand = torch.max(nn.Softmax(dim=1)(outputs_rand).data, 1)
            correct_rand += predicted_rand.eq(rand_labels.data).cpu().sum()

#             #save results on 0th iteration
#             if i == 0:
#                 validloss, validacc = test(validloader, model, bn_eval=False)
#                 testloss, testacc = test(testloader, model, bn_eval=False)

#                 validloss_all.append(validloss)
#                 validacc_all.append(validacc)
#                 testloss_all.append(testloss)
#                 testacc_all.append(testacc)
#                 trainloss_all.append(loss.item())
#                 trainacc_all.append(correct.numpy() / total * 100)
#                 print('[%d, %5d] train loss: %.3f, train acc: %.3f, val loss: %.3f, val acc: %.3f, test loss: %.3f, test acc: %.3f' %
#                       (epoch + 1, i + 1, loss.item(), correct.numpy() / total * 100, validloss, validacc, testloss, testacc))

            # print statistics
            running_loss += loss.item()
            running_loss_true += true_loss.item()
            running_loss_rand += rand_loss.item()
            
            if i % args.print_every == args.print_every -1:   
                validloss, validacc = test(validloader, model, bn_eval=False)
                testloss, testacc = test(testloader, model, bn_eval=False)

                validloss_all.append(validloss)
                validacc_all.append(validacc)
                testloss_all.append(testloss)
                testacc_all.append(testacc)
                trainloss_all.append(running_loss / args.print_every)
                trainacc_all.append(correct.numpy() / total * 100)
                print('[%d, %5d] rand loss: %.3f, true loss: %.3f, loss: %.3f, train acc: %.3f, rand_train_acc: %.3f, val loss: %.3f, val acc: %.3f, test loss: %.3f, test acc: %.3f, train_true is %s, gamma is %.2f' %
                      (epoch + 1, i + 1, 
                       running_loss_rand / args.print_every, running_loss_true / args.print_every, 
                       running_loss / args.print_every, 
                       correct.numpy() / total * 100, correct_rand.numpy() / total * 100, validloss, validacc, testloss, testacc, train_true, args.gamma))
                

                wandb.log({'Epoch': epoch + 1, 'Iter': i+1, 'Random Train Loss': running_loss_rand / args.print_every, 'Random Train Accuracy': correct_rand.numpy() / total * 100, 'True Train Loss': running_loss_true / args.print_every, 'Actual Train Loss': running_loss / args.print_every, 'True Train Accuracy': correct.numpy() / total * 100, 'Valid Loss': validloss, 'Valid Accuracy':validacc, 'Test Loss': testloss, 'Test Accuracy': testacc, 'Gamma':args.gamma})


                running_loss = 0.0
                running_loss_true = 0.0
                running_loss_rand = 0.0
                correct = 0
                correct_rand=0
                total = 0

            iter_counter += 1
            if iter_counter >= args.num_iter_true and train_true:
                train_true = False
                iter_counter = 0

            if iter_counter >= args.num_iter_rand and not train_true:
                train_true = True
                iter_counter = 0
                
            if i % args.decay_gamma_every == 0:
                args.gamma *= args.decay_gamma_rate

        if epoch % args.save_every==0:
            outfile = os.path.join(workdir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state_dict':model.state_dict()}, outfile)
    
    #save final model
    outfile = os.path.join(workdir, 'final_model.tar')
    torch.save({'epoch':epoch, 'state_dict':model.state_dict()}, outfile)

    return trainloss_all, trainacc_all, validloss_all, validacc_all, testloss_all, testacc_all
       
        
def main():    
    parser = make_parser()
    args = parser.parse_args()
    
    if args.mrl is not None:
        multi_rand_layer = [int(x) for x in args.mrl.split(',')]
    else:
        multi_rand_layer = []
    
    #set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#     random.seed(args.seed)
    np.random.seed(args.seed)
    
    #model name creation
    model_run_name=args.train_data
    if args.test_data is not None:
        model_run_name = model_run_name + '_' + args.test_data
    else:
        model_run_name = model_run_name + '_origtest'
    model_run_name = model_run_name + '_' + args.model_choice
    if args.train_true:
        model_run_name = model_run_name + '_trainfirst'
    else:
        model_run_name = model_run_name + '_randfirst'
    model_run_name = model_run_name + '_iterT' + str(args.num_iter_true) + '_iterR' + str(args.num_iter_rand) + '_iterRsb' + str(args.num_iter_rand_sb)
    model_run_name = model_run_name + '_gam' + str(args.gamma) + '_alf' + str(args.alpha)
    model_run_name = model_run_name + '_lrT' + str(args.lr_true) + '_lrR' + str(args.lr_rand)
    if args.mrl is not None:
        model_run_name = model_run_name + '_mrl' + str(args.mrl)
    if args.decay_gamma_rate != 1.0:
        model_run_name = model_run_name + '_dgm' + str(args.decay_gamma_rate) + 'per' + str(args.decay_gamma_every)
    if args.train_label_noise > 0:
        model_run_name = model_run_name + '_ln' + str(args.train_label_noise)
    model_run_name = model_run_name + '_ep' + str(args.num_epochs)
    if args.simultaneous:
        model_run_name = model_run_name + '_simt'
    if args.train_separately:
        model_run_name = model_run_name + '_sep'
    if args.consistent_rand:
        model_run_name = model_run_name + '_csist'
    if args.pos_rand:
        model_run_name = model_run_name + '_posr'
    if args.random_per_batch:
        model_run_name = model_run_name + '_rpb'
    if args.train_rand_true:
        model_run_name = model_run_name + '_tforr'
    model_run_name = model_run_name + '_seed' + str(args.seed)
    model_run_name = model_run_name + '_inittype' + args.init_type
    if args.model_path is not None:
        model_run_name = model_run_name + '_fromexist'
    else:
        model_run_name = model_run_name + '_fromfresh'
        
        
    if len(args.group_vars) > 0:
        if len(args.group_vars) == 1:
            group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
        else:
            group_name = args.group_vars[0] + str(getattr(args, args.group_vars[0]))
            for var in args.group_vars[1:]:
                group_name = group_name + '_' + var + str(getattr(args, var))
        wandb.init(project="random_game",
               group=args.model_name,
               name=group_name)
        for var in args.group_vars:
            wandb.config.update({var:getattr(args, var)})
            
    workdir = os.path.join(args.save_path, model_run_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    print('saving to workdir %s' % workdir)
    
    #load data
    trans = ([transforms.ToTensor()])
    trans = transforms.Compose(trans)
    
    if args.train_data == 'mnist':
        fulltrainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=trans)
    if args.train_data == 'cifar10':
        fulltrainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=trans)
        
    train_split, val_split = int(len(fulltrainset)*0.9), int(len(fulltrainset)*0.1)
    train_set, valset = torch.utils.data.random_split(fulltrainset, [train_split, val_split])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=4, pin_memory=True)
    
    #add label noise to train if needed
    if args.train_label_noise > 0:
        num_random = int(train_split * args.train_label_noise)
        train_random_labels = torch.randint(0, args.num_classes, (num_random,))
        train_random_idx = np.random.choice(trainloader.dataset.indices, num_random)
        trainloader.dataset.dataset.targets[train_random_idx] = train_random_labels

    rand_labels_raw = torch.randint(0, args.num_classes, fulltrainset.targets.shape)
    rand_labels_train, rand_labels_val = torch.utils.data.random_split(rand_labels_raw, [train_split, val_split])

    rand_labels_train = torch.utils.data.DataLoader(rand_labels_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    rand_labels_val = torch.utils.data.DataLoader(rand_labels_val, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.test_data is not None:
        if args.train_data == 'mnist':
            test_set = MNISTCDataset(root=os.path.join(data_path, args.test_data), train=False, transform=trans)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    else:
        if args.train_data == 'mnist':
            test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=trans)
        elif args.train_data == 'cifar10':
            test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=trans)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)    
    
    
    #initialize models
    if args.model_choice == 'lenet':
        if args.pos_rand:
            model = RandomGamePos(model_func=LeNet, num_class = args.num_classes)
        else:
            model = RandomGame(model_func=LeNet, num_class = args.num_classes, multi_rand_layer=multi_rand_layer)
            
    elif args.model_choice == 'convnet':
        model = RandomGame(model_func=MnistConvNet, num_class = args.num_classes, multi_rand_layer=multi_rand_layer)
        
    if args.model_path is not None:
        model_checkpoint = torch.load(args.model_path)
        model.load_state_dict(model_checkpoint['state_dict'])
    
    if args.mse:
        mse_loss_fn = nn.MSELoss(reduction='mean')
    
    

    # Optimizers
    true_parameters = list(model.feature.parameters()) + list(model.true_classifier.parameters())
    optimizer_T = torch.optim.SGD(true_parameters, lr = args.lr_true, weight_decay=args.l2_true)
    optimizer_R = torch.optim.SGD(model.rand_classifier.parameters(), lr=args.lr_rand, weight_decay=args.l2_rand)
    
    if args.train_separately:
        optimizer_E = torch.optim.SGD(model.feature.parameters(), lr = args.lr_true, weight_decay=args.l2_true)
        optimizer_TC = torch.optim.SGD(model.true_classifier.parameters(), lr = args.lr_true, weight_decay=args.l2_true)
    else:
        optimizer_E=None
        optimizer_TC=None
    
    if args.save_init_model:
        outfile = os.path.join(workdir, 'init_model.tar')
        torch.save({'epoch':-1, 'state_dict':model.state_dict()}, outfile)
                              
    if use_cuda:
        model.cuda()

    trainloss_all, trainacc_all, validloss_all, validacc_all, testloss_all, testacc_all = train(args, model, 
            optimizer_T, optimizer_R, trainloader, validloader, testloader, rand_labels_train, 
            workdir, mse_loss_func=None, optimizer_E=optimizer_E, optimizer_TC=optimizer_TC)
    
    #save metrics
    with h5py.File(os.path.join(workdir, 'perf_metrics.h5'), 'w') as f:
        f.attrs['print_every'] = args.print_every
        f.attrs['num_epochs'] = args.num_epochs
        f.attrs['keys'] = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']

        tl_dset = f.create_dataset('train_loss', np.array(trainloss_all).shape)
        tl_dset[:] = np.array(trainloss_all)
        ta_dset = f.create_dataset('train_acc', np.array(trainacc_all).shape)
        ta_dset[:] = np.array(trainacc_all)
        vl_dset = f.create_dataset('val_loss', np.array(validloss_all).shape)
        vl_dset[:] = np.array(validloss_all)
        va_dset = f.create_dataset('val_acc', np.array(validacc_all).shape)
        va_dset[:] = np.array(validacc_all)
        el_dset = f.create_dataset('test_loss', np.array(testloss_all).shape)
        el_dset[:] = np.array(testloss_all)
        ea_dset = f.create_dataset('test_acc', np.array(testacc_all).shape)
        ea_dset[:] = np.array(testacc_all)
    wandb.run.finish()
                              
if __name__ == '__main__':
    main()