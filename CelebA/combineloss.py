import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#no morm
class CombineLossV1(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, globalfea, maskweight, clsweight, target):
        losses = 0
        for i in range(0,40):#共40个属性
            selectmask = maskweight[i,:]
            attributefea = torch.mul(globalfea, selectmask)
            selectClsweight = clsweight[:,i*2:i*2+2]
            predict = attributefea.mm(selectClsweight)
            groundTruth = target[:,i]
            groundTruth = groundTruth.long()
            loss = self.ce(predict, groundTruth)
            losses += loss
        return losses

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2)
    w1 = torch.norm(x1, dim=1)
    w2 = torch.norm(x2, dim=1)
    return ip / torch.ger(w1,w2).clamp(min=eps)


#feature norm
class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.s = np.log((2.0**126)/2)/2

    def forward(self, globalfea, maskweight, clsweight, target):
        losses = 0
        for i in range(0,40):#共40个属性
            selectmask = maskweight[i,:]
            #print(selectmask)
            attributefea = torch.mul(globalfea, selectmask)
            #attributefea = F.dropout(attributefea, p=0.5)
            #print(attributefea.size())
            attributefeaNorm = torch.norm(attributefea, dim=1)
            #print(attributefeaNorm.size())
            attributefea = (torch.div(attributefea, attributefeaNorm.view(-1,1)))* self.s
            #print(attributefea.size())
            selectClsweight = clsweight[:,i*2:i*2+2]
            #print(selectClsweight.size())
            selectClsweightNorm = torch.norm(selectClsweight, dim=0)
            selectClsweight = torch.div(selectClsweight, selectClsweightNorm)
            #print(selectClsweight)

            predict = attributefea.mm(selectClsweight)
            groundTruth = target[:,i]
            groundTruth = groundTruth.long()
            loss = self.ce(predict, groundTruth)
            losses += loss
        return losses