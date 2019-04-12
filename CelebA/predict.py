'''
    implement training process for Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from PIL import Image, ImageEnhance
from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from load_imglist import ImageList
from resnet import resnet18
from vgg import vgg16, vgg16_bn
from MobileNetV2 import MobileNetV2Layers
#model = LightCNN_9Layers(num_classes=40)
#model = resnet18()
#model = MobileNetV2Layers()
model = vgg16_bn()
model = model.cuda()
checkpoint = torch.load('F:/celeba/test/results/lightcnn/normcls/lightCNN_46_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

model.eval()
imgRoot = 'F:/celeba/img_align_celeba/'
inFile = open('F:/celeba/test/testlabel.txt','r')

outFile = open('F:/celeba/test/results/lightcnn/normcls/testpredict.txt','w')
transform = transforms.Compose([transforms.CenterCrop((202, 162)),transforms.ToTensor(),])

allLines = inFile.readlines()
for line in allLines:
    imgName = line.split(' ')[0]
    outFile.write(imgName + ' ')
    imgPath = imgRoot + imgName
    print(imgPath)
    img = Image.open(imgPath)
    img = transform(img)
    input = img.unsqueeze(0).to('cuda')
    globalFea, mWeight, cWeight = model(input)

    for i in range(0, 40):  # 共40个属性
        selectmask = mWeight[i, :]
        attributefea = torch.mul(globalFea, selectmask)
        attributefeaNorm = torch.norm(attributefea, dim=1)
        # print(attributefeaNorm.size())
        attributefea = (torch.div(attributefea, attributefeaNorm.view(-1, 1))) * np.log((2.0**126)/2)/2
        # print(attributefea.size())
        selectClsweight = cWeight[:, i * 2:i * 2 + 2]
        # print(selectClsweight.size())
        selectClsweightNorm = torch.norm(selectClsweight, dim=0)
        selectClsweight = torch.div(selectClsweight, selectClsweightNorm)
        # print(selectClsweight)

        predict = attributefea.mm(selectClsweight)
        predLabel = torch.max(predict, 1)[1].data.cpu().numpy()[0]
        if predLabel == 0:
            outFile.write('1' + ' ')
        else:
            outFile.write('-1' + ' ')
    outFile.write('\n')

    # for i in range(0, 40):  # 共40个属性
    #     selectmask = mWeight[i, :]
    #     attributefea = torch.mul(globalFea, selectmask)
    #     selectClsweight = cWeight[:, i * 2:i * 2 + 2]
    #     predict = attributefea.mm(selectClsweight)
    #     predLabel = torch.max(predict, 1)[1].data.cpu().numpy()[0]
    #     if predLabel==0:
    #         outFile.write('1' + ' ')
    #     else:
    #         outFile.write('-1' + ' ')
    # outFile.write('\n')
    #print(attributes)