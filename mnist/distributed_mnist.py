#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 02:17:45 2020

@author: liujiachen
"""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import time

import torch.nn.parallel

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data 
import torch.utils.data.distributed
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import timedelta
#import torch.nn.DistributedDataParallelCPU

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=0)


def train(epoch,model,optimizer,args,train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test(model,args,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--init-method', type=str, default='tcp://10.10.1.2:23456')
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--world-size',type=int, default=0)
    
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #初始化
    
    dist.init_process_group(init_method=args.init_method,timeout=timedelta(seconds=1800), \
                            backend="gloo", world_size=args.world_size,rank=args.rank,group_name="pytorch_test")
    #dist.init_process_group(init_method='tcp://10.10.1.2:23256',backend="gloo", world_size=1,rank=0,group_name="pytorch_test")
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    train_dataset=datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    # 分发数据
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,args.world_size,rank=args.rank)
        
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=(train_sampler is None),\
                                               sampler=train_sampler,batch_size=args.batch_size, **kwargs)
    # train_loader = torch.utils.data.DataLoader(train_dataset,drop_last=True,batch_size=int(args.batch_size), **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # device = torch.device("cuda" if use_cuda else "cpu")
    model = Net() 
    print("Discover {} GPU".format(torch.cuda.device_count()))
    if args.cuda:
        # 分发模型
        #model.cuda()
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids =args.rank)
        model = torch.nn.DataParallel(model,device_ids=args.rank).cuda()
        model.cuda()
        print("DistributedDataParallel GPU")
    else:
        print("DistributedDataParallel CPU")
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    tot_time=0;

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        start_cpu_secs = time.time()
        #long running
        train(epoch,model,optimizer,args,train_loader)
        end_cpu_secs = time.time()
        print("Epoch {} of {} took {:.3f}s".format(
            epoch , args.epochs , end_cpu_secs - start_cpu_secs))
        tot_time+=end_cpu_secs - start_cpu_secs
        test(model,args,test_loader)
    
    print("Distribute on {} workers use = {:.3f}s".format(args.world_size, tot_time))
    
    
    dist.destroy_process_group()
    
if __name__ == '__main__':
    main()
