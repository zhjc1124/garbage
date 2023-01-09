# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import torch as t
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision as tv
import numpy as np
from torchvision.transforms import ToPILImage

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from PIL import Image, ImageDraw, ImageFont
import cv2
use_gpu = t.cuda.is_available()

if use_gpu == True:
    print('gpu可用')
else:
    print('gpu不可用')


epochs = 50  # 训练次数
batch_size = 6  # 批处理大小
num_workers = 0  # 多线程的数目
model = 'model.pt'   # 把训练好的模型保存下来



# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 在训练集中，shuffle必须设置为True，表示次序是随机的
trainset = datasets.ImageFolder(root='datasets/train/', transform=data_transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# 在测试集中，shuffle必须设置为False，表示每个样本都只出现一次
testset = datasets.ImageFolder(root='datasets/test/', transform=data_transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('cardboard', 'glass', 'metal', 'paper','plastic','trash')

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        #Sequential 是一个特殊Module, 包含几个子module,前向传播时会将输入一层一层的传递下去
        self.left = nn.Sequential(
                #卷积层
                nn.Conv2d(inchannel,outchannel,3,stride, 1,bias=False),
                #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
                #这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
                nn.BatchNorm2d(outchannel),
                #激活函数采用ReLU
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut
        
    #前向传播
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=6):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                #最大池化
                nn.MaxPool2d(3, 2, 1))
        
        # 重复的layer，分别有3，4，6，3个residual block
        #共四层
        self.layer1 = self._make_layer( 64, 64, 3)
        self.layer2 = self._make_layer( 64, 128, 4, stride=2)
        self.layer3 = self._make_layer( 128, 256, 6, stride=2)
        self.layer4 = self._make_layer( 256, 512, 3, stride=2)

        #分类用的全连接
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        
        #构建layer,包含多个residual block
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #平均池化
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if use_gpu:
    net = t.load(model)
else:
    net = t.load(model, 'cpu')
net.eval()

def eval_img(img_path):
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        img = Image.open(img_path).convert('RGB')
        image = data_transform(img).unsqueeze(0)
        if use_gpu:
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        outputs = net(image)
        _, predicted = t.max(outputs, 1)
        print('检测到类别为：', classes[predicted])
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # cv2.putText(img, classes[predicted], (192, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # cv2.imshow('20201003731汪鑫', img)
        plt.figure('20201003731汪鑫') 
        plt.title('Detect: ' + classes[predicted]) 
        plt.axis('off')
        plt.imshow(img)
        plt.show()

import tkinter as tk
from tkinter import filedialog

def upload_file():
    selectFile = tk.filedialog.askopenfilename() 
    eval_img(selectFile)

root = tk.Tk()

root.title('20201003731汪鑫')
frm = tk.Frame(root)
frm.grid(padx='20', pady='30')
btn = tk.Button(frm, text='上传图片', command=upload_file)
btn.grid(row=0, column=0, ipadx='3', ipady='3', padx='10', pady='20')
entry1 = tk.Entry(frm, width='40')
entry1.grid(row=0, column=1)

root.mainloop()