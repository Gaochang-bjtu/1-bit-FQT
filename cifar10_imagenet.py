import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import bnn.models.resnet1 as models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import time
from bnn.engine import BinaryChef

from bnn import BConfig, prepare_binary_model, Identity
from  bnn.models.resnet_imagenet import resnet18
from bnn.models.vgg import VGG
from examples.utils import AverageMeter, ProgressMeter, accuracy
from data.loader import load
import log
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--print_freq', type=int, default=100,
                    help='logs printing frequency')
parser.add_argument('--out_dir', type=str, default='')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--optimizer', default= 'Adam', type = str)
parser.add_argument('--pretrained', default= 'true', type= str)
parser.add_argument('--dataset', default= 'cub', type= str)
parser.add_argument('--gradient', default= 'True', type= str)
parser.add_argument('--type', default= 'QAT', type= str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding =16),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='/data/datasets/cifar10/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='/data/datasets/cifar10/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='/data/datasets/cifar-100/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='/data/datasets/cifar-100/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)    
else:
    trainloader, testloader = load(args.dataset, 128, 256)
logger = log.get_logger(os.path.join('./checkpoint/', args.optimizer + args.type + args.gradient + args.dataset + '_logger.log'))

# Model
print('==> Building model..')
#net = resnet18(num_classes = 10)

net = models.__dict__['resnet18'](stem_type= 'basic', num_classes=200, block_type=models.PreBasicBlock, activation=nn.PReLU)


#net = VGG('VGG16', 200)
bchef = BinaryChef('examples/recepies/imagenet-baseline.yaml')
model = bchef.run_step(net, 1)
logger.info(args)
#logger.info(model)
#net = torch.nn.DataParallel(net)
if args.pretrained == 'true':
    print('##############################################')
    #checkpoint = torch.load('./checkpoint/ckpt_full.pth')
    #checkpoint = checkpoint['net']
    #checkpoint = torch.load('./checkpoint/resnet_imagenet/ckpt_epoch{}.pth'.format(20))
    #net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})

    pre_weights = torch.load("./experiments/model_best.pth.tar", map_location='cpu')
    pre_weights = pre_weights['state_dict']
    #for k,v in pre_weights.items():
    #    print(k)
    #for k,v in net.state_dict().items():
    #    print(k)
    pre_dict = {k.replace('module.', ''): v for k, v in pre_weights.items() if net.state_dict()[k.replace('module.', '')].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    print(missing_keys, unexpected_keys)

#checkpoint = torch.load('./checkpoint/ckpt_full.pth')
#net.load_state_dict(checkpoint['net'])
#net = net.module()
# Binarize
print('==> Preparing the model for binarization')
#net = nn.DataParallel(model, device_ids=[0, 1, 2])
print(net)

net = net.to(device)
if 'cuda' in device:
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'Adam':

    conv_lr = 1e-5
    optimizer = optim.Adam([
    {'params': net.fc.parameters(), 'lr': 1e-3},
    {'params': net.layer1.parameters(), 'lr': conv_lr},
    {'params': net.layer2.parameters(), 'lr': conv_lr},
    {'params': net.layer3.parameters(), 'lr': conv_lr},
    {'params': net.layer4.parameters(), 'lr': conv_lr},
    {'params': net.bn1.parameters(), 'lr': conv_lr},
    {'params': net.conv1.parameters(), 'lr': conv_lr}

    ], weight_decay=1e-5)
    '''
    optimizer = optim.Adam([
    {'params': net.classifier.parameters(), 'lr': 1e-3},
    {'params': net.features.parameters(), 'lr': 1e-5}

    ], weight_decay=1e-5)
    '''
else:
    optimizer = optim.SGD([
    {'params': net.fc.parameters(), 'lr': 1e-3},
    {'params': net.layer1.parameters(), 'lr': 1e-4},
    {'params': net.layer2.parameters(), 'lr': 1e-4},
    {'params': net.layer3.parameters(), 'lr': 1e-4},
    {'params': net.layer4.parameters(), 'lr': 1e-4},
    {'params': net.bn1.parameters(), 'lr': 1e-4},
    {'params': net.conv1.parameters(), 'lr': 1e-4}

    ], momentum=0.9, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 140, 180], 0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
net = nn.DataParallel(net, device_ids=[0]).cuda()
# Training
def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    print('\nTrain Epoch: %d' % epoch)
    net.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, = accuracy(outputs, targets)

        top1.update(acc1.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)
        #exit(0)
    logger.info("=>Top1: {:.3f}, loss: {:.3f}".format(top1.avg, losses.avg))

def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, data_time, losses, top1],
        prefix="Test Epoch: [{}]".format(epoch))

    global best_acc
    net.eval()
    end = time.time()
    #torch.save(net.state_dict(), './checkpoint/resnet_cifar10/ckpt{}.pth'.format(epoch))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets)

            top1.update(acc1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/flowers_ckpt2.pth')
        best_acc = acc
    print('Current acc: {}, best acc: {}'.format(acc, best_acc))
    logger.info("=>Top1: {:.3f}, loss: {:.3f}".format(top1.avg, losses.avg))

for epoch in range(start_epoch, start_epoch+60):
    train(epoch)
    scheduler.step()
    test(epoch)
print(args.pretrained, args.optimizer, args.dataset)
