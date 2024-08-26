import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import sys
sys.path.append('../')
sys.path.append('../csrc/binop')
import time
import itertools
import numpy as np

from tqdm import tqdm
tqdm.monitor_interval = 0

# import models as models
import Bin_VGG
import VGG
import binop
from util import binop_train
from util import bin_save_state
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


os.environ["MKL_NUM_THREADS"] = "1"
from resnet import resnet18






def test(model, evaluate=False):
    model.eval()
    pbar = tqdm(test_loader, total=len(test_loader))
    input = torch.ones((128, 3, 32, 32), dtype= torch.float, requires_grad= True)
    #input = torch.ones((256, 10000), dtype= torch.float, requires_grad= True)
    for data, target in pbar:
        
        start_time = time.time()
        output = model(input)
        middle_time = time.time()
        #loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        
        loss = output.sum()
        
        loss.backward()
        end_time = time.time()
        print('forward_time:', middle_time - start_time, 'backward_time:', end_time - middle_time)
        






if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch Cifar-10')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--lr-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to decay the lr (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='whether to run evaluation')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    target_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(args)
    cnt = 0
    lr_cnt = 0

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # load data
    kwargs = {'num_workers': 0, 'pin_memory': False}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/data/datasets/cifar10/', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/data/datasets/cifar10/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    best_acc = 0.0
    model_ori = None
    model_train = None
    model_test = None

    #model_test = Bin_VGG.Bin_linear_test()
    #model_ori = Bin_VGG.linear_test()
    #model_test = Bin_VGG.Bin_conv2d_test()
    #model_ori = Bin_VGG.Conv2d_test()
    model_test = Bin_VGG.Bin_VGG_train('VGG16')
    #model_ori = VGG.VGG('VGG16')
    #model_test = resnet18()
    i = 0
    if args.evaluate == False:
        start_test = time.time()
        for i in range(1):
            if model_ori is None:
                test(model_test, evaluate=False)
            else:
                test(model_ori, evaluate=False)
        end_test = time.time()
        seconds = end_test - start_test
        #m, s = divmod(seconds, 60)
        #h, m = divmod(m, 60)
        print("Test Total Time: %02d" % (seconds))
        

    


