import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from torch import cat
from bert import BERT
from torch.autograd import Variable
import numpy as np
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

classes = 10

model_urls = {
    #'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    #'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    #'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    #'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    #'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                   
               
            
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(9 * 1024, 4096))
        self.fc7.add_module('relu7', nn.ReLU(inplace = True))
        self.fc7.add_module('drop7', nn.Dropout(p = 0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(512, classes))
        #self.apply(weights_init)
        
        self.attention_pooling = BERT(4096, hidden=512, n_layers=3, attn_heads=32)
        
        self.pos_embed = nn.Embedding(10, 512) # position embedding

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

#     def single_forward(self, x):
        
#         #x = self.fc(x)

#         return x
    
    def _forward(self, x):
        B, T, C, H, W = x.size()
        #print(B,T,C,H,W)
        x = x.transpose(0, 1)
        context = Variable(Tensor(np.random.normal(0, 1, (B, 1, 512)),device = x.device))
        
        x_list = [context]
        for i in range(9):
            z = self.conv1(x[i])
            z = self.bn1(z)
            z = self.relu(z)
            z = self.maxpool(z)

            z = self.layer1(z)
            z = self.layer2(z)
            #z = self.layer3(z)
     #       z = self.layer4(z)
        #    print(z.shape)
            z = self.avgpool(z)
            z = torch.flatten(z, 1)
#             z = self.conv(x[i])
         #   print(z.shape)
            #z = self.fc6(z.view(B, -1))
#            print(z.shape)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x_list = cat(x_list, 1)
        
        seq_len = T+1
        #print(seq_len)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        #print(pos.shape)
        pos = pos.unsqueeze(0).repeat(B,1)
        #print(pos.shape)
        pos = self.pos_embed(pos)
        #print(pos.shape)
            
        positions = np.arange(9)
        query_ind = np.sort(np.random.choice(9,3,replace=False)) + 1
        context_ind = np.array([pos for pos in (positions+1) if pos not in query_ind])
        context_ind = np.append(np.array([0]),context_ind)
#         print(query_ind)
#         print(context_ind)
        #x_list = np.array(x_list)
        context = x_list[:,context_ind]
        
        query = x_list[:,query_ind]
        #x = cat(context, 1)
        #x = self.fc7(x.view(B, -1))
        x = self.attention_pooling.forward(context)
        global_context = x[:,0]
        
        choices_ind = [np.random.choice(3) for i in range(B)]#, dtype = torch.long)
     
        choices = torch.zeros((B,x_list.shape[1],x_list.shape[2]),device=x.device)
        #print(choices_ind)
        for i in range(len(choices_ind)):
            zero = torch.zeros((x_list.shape[1],x_list.shape[2]))
            #print(zero.shape)
            #print(zero)
            zero[choices_ind[i]] = 1
            #print(zero)
            #print(choices[i].shape)
            choices[i] = zero
            
        #print(choices.shape)
        #print(pos.shape)
        pos_random = torch.sum(pos*choices,dim=1)
        #print(pos_random.shape)
        #print(global_con
        global_context = global_context + pos_random#pos[:,query_ind[choice]]

        final = torch.squeeze(torch.matmul(torch.unsqueeze(global_context,1),torch.transpose(query,1,2)),1)
        
#         x = self.classifier(x[:,0])
        #print(final.shape,B)
        #print(np.array(choice).shape)
        #print(torch.from_numpy(np.array(choice)).shape)
#         print(torch.from_numpy(np.array(choice)).shape)
        #print(torch.unsqueeze(torch.Tensor(torch.Tensor(choice),device = x.device),0).shape)
        final_choice = torch.from_numpy(np.array(choices_ind)).to(x.device)#.repeat(B)
        #print(final_choice.shape)
        return final, final_choice.type(torch.long)

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
   # if pretrained:
 #       state_dict = load_state_dict_from_url(model_urls[arch],
                                             # progress=progress)
  #      model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    #modules=list(model.children())[:-3]
    #model=nn.Sequential(*modules)
    return model

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
