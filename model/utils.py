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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from numpy.core.arrayprint import printoptions
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import sys
from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def to_one_hot_3d(tensor, n_classes): #shape = [batch, s, h, w]-> [batch, s, h, w, c]-> [batch, c, h, w]
    b, s, h, w = tensor.size()
    print(tensor.size())
    if n_classes == 2:
        tensor1, tensor2 = torch.clone(tensor), torch.clone(tensor)
        tensor1[tensor1 == 0]  = 1.0
        tensor2[tensor2 == 1]  = 0.0
        tensor1, tensor2 = tensor1.unsqueeze(-1), tensor2.unsqueeze(-1)

        one_hot = torch.cat((tensor1, tensor2), -1)
        one_hot = one_hot.squeeze(0)
        one_hot = one_hot.permute(3,0,1,2)
    return one_hot

