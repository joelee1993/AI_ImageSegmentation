import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import torchvision 
import torch.utils.data as data
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from numpy.core.arrayprint import printoptions

class BCEDiceLoss(nn.Module):
    def __init__(self, a, b):
        super(BCEDiceLoss, self).__init__()
        self.a = a
        self.bce = nn.BCEWithLogitsLoss()
        self.b = b
        self.dice = DiceLoss()
    def forward(self, input, target):
        inputs = F.sigmoid(input)
        inputs = inputs.view(-1)
        targets = target.view(-1)  
        return torch.clamp((self.a*self.bce(inputs,targets)+self.b*self.dice(inputs, targets)),0,1)
        


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, smooth =1 ):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = target.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum()+targets.sum()+smooth)
        return torch.clamp((1-dice),0,1)

class FocalLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, inputs, target, smooth =1 ):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = target.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum()+targets.sum()+smooth)
        return torch.clamp((1-dice),0,1)
