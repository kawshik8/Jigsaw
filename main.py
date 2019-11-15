import argparse
import os
from dataload import DataLoader
import torch
import torch.nn as nn
from network import Network
#from resnet import resnet50
from resnet import ResNet, resnet50
#from resnet import resnet50
import numpy as np
import time
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchsummary import summary
import torch.nn as nn
from initialize import initialize_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type = str, default = '../checkpoints', help = 'path for saving trained models')
parser.add_argument('--image_dir', type = str, default = '../data', help = 'directory for resized images')
parser.add_argument('--log_step', type = int, default = 100, help = 'step size for prining log info')
parser.add_argument('--save_step', type = int, default = 100, help = 'step size for saving trained models')

# Model parameters
parser.add_argument('--num_epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--num_workers', type = int, default = 4)
parser.add_argument('--learning_rate', type = float, default = 1e-3)
args = parser.parse_args()
print(args)

initialize_data(args.image_dir)

train_set = DataLoader(args.image_dir + "/train")
val_set = DataLoader(args.image_dir + "/val")

train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True,num_workers = args.num_workers)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size, shuffle = False,num_workers = args.num_workers)

model = resnet50().to(device)
        
print(summary(model, input_size=(9, 3, 8, 8)))

# model.load_state_dict(torch.load('/mnt/cephfs/lab/wangyuqing/jiasaw/model/imagenet_models/model-6-100.ckpt'))

criterion = nn.CrossEntropyLoss().to(device)
        
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr = args.learning_rate)
# total_step = len(data_loader) 
        
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

last_time=0
##############
# def main(args):

        
        ################
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
    
    
#     last_time=0
#     for epoch in range(args.num_epochs):
#     #    try: 
           
#             for i, (images, targets) in enumerate(train_loader):
#                 #print(i,(images).shape,(targets).shape,(original).shape)
# #                 if torch.cuda.available():
#                 images=images.to(device)
#                 targets=targets.to(device)
#                 correct=0
#                 #print(len(targets),targets)#,len(targets[0]))
#                 #print(len(images),len(images[0]),len(images[0][0]),len(images[0][0][0]),len(images[0][0][0][0]))
#                 outputs = model(images)
#                 #print(outputs.shape)
# #                 if torch.cuda.available():
#                 loss = criterion(outputs, targets).to(device)
#                 pred = outputs.data.max(1, keepdim= True)[1]
#                 correct+= pred.eq(targets.data.view_as(pred)).cpu().sum()
# #                 else:
# #                     loss = criterion(outputs, targets)
#                 model.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 # Print log info
#                 this_time=time.time()-last_time
#                 last_time=time.time()
#                 if i % args.log_step == 0:
#                     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}, Time: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(),(100. * correct / args.batch_size), this_time))
#                # Save the model checkpoints
#                 if (i + 1) % args.save_step == 0:
#                     torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
#      #   except:
      #      pass
        
        #########################

def train(epoch):
    model.train()
    total_step = len(train_loader)
    for i, (images, targets) in enumerate(train_loader):
        correct = 0
        images=images.to(device)
        targets=targets.to(device)
        outputs = model(images)

        loss = criterion(outputs, targets).to(device)
        pred = outputs.data.max(1, keepdim= True)[1]
        correct+= pred.eq(targets.data.view_as(pred)).cpu().sum()

        model.zero_grad()
        loss.backward()
        optimizer.step()
                # Print log info

#         this_time=time.time()-last_time
#         last_time=time.time()
        if i % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(),(100. * correct / args.batch_size)))
       # Save the model checkpoints

            ################# 
        
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    for i, (images, targets) in enumerate(val_loader):
        images=images.to(device)
        targets=targets.to(device)
        outputs = model(images)

        validation_loss += criterion(outputs, targets)#.to(device)#
        pred = outputs.data.max(1, keepdim= True)[1]
        correct+= pred.eq(targets.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / (len(val_loader))
    validation_loss /= len(val_loader)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader),
        acc))
    if epoch == args.epochs:
        print(epoch)
    return acc,validation_loss
     



  ####################
        
        
if __name__ == '__main__':
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    val_loss = 200
    val_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        train(epoch)
        va,vl = validation()
        scheduler.step(vl)
        print(va,vl)
        if (i + 1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))        

#         if va > val_acc:
#             val_acc = va
#             model_file = 'model_acc_' + str(epoch) + '.pth'
#             torch.save(model.state_dict(), model_file)
#             print('\nSaved Best acc model to ' + model_file +'\n')  

#         if vl < val_loss:
#             val_loss = vl
#             model_file = 'model_loss_' + str(epoch) + '.pth'
#             torch.save(model.state_dict(), model_file)
#             print('\nSaved Best loss model to ' + model_file + '\n')


