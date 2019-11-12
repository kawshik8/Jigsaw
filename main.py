import argparse
import os
from dataload import DataLoader
import torch
import torch.nn as nn
from network import Network
import numpy as np
import time
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchsummary import summary

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    data_set = DataLoader(args.image_dir + "/train")#datasets.ImageFolder(args.data + "/train", transform = transforms))#DataLoader(args.image_dir)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = args.batch_size, shuffle = True,num_workers = args.num_workers)
    model = models.resnet50(pretrained=False)
    modules=list(model.children())[:-3]
    model=nn.Sequential(*modules)
    print(summary(model, input_size=(3, 8, 8)))
    #model=Network()#.cuda()#nn.DataParallel(Network()).cuda()
    #model.load_state_dict(torch.load('/mnt/cephfs/lab/wangyuqing/jiasaw/model/imagenet_models/model-6-100.ckpt'))
    criterion = nn.CrossEntropyLoss().cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)
    total_step = len(data_loader)
    last_time=0
    for epoch in range(args.num_epochs):
        try: 
            for i, (images, targets, original) in enumerate(data_loader):
                images=images#.cuda()
                targets=targets#.cuda()
                outputs = model(images)
                loss = criterion(outputs, targets).cuda()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                # Print log info
                this_time=time.time()-last_time
                last_time=time.time()
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(),this_time))
               # Save the model checkpoints
                if (i + 1) % args.save_step == 0:
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = './checkpoints', help = 'path for saving trained models')
    parser.add_argument('--image_dir', type = str, default = './data', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 100, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    args = parser.parse_args()
    print(args)
    main(args)