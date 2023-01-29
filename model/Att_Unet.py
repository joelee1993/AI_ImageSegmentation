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
from utils import to_one_hot_3d


class CTDataset(Dataset):
    def __init__(self, CT_image_root, MRI_label_root): #transform = None):
        #----------------------------------------
        # Initialize paths, transforms, 
        #----------------------------------------
        self.CT_path = CT_image_root
        self.MRI_path = MRI_label_root
        self.CT_name = os.listdir(os.path.join(CT_image_root))
        #self.transform = transform
        self.MRI_name = os.listdir(os.path.join(MRI_label_root))
        
    def __getitem__(self, index):
        #----------------------------------------
        #1. Read from file(using sitk.ReadImage)
        #2. Preprocess the data(torchvision.Transform)
        #3. Return the data (e.g. image and label)
        #----------------------------------------
        # Select the sample
        CT_ID = self.CT_name[index]
        print(CT_ID)
        MRI_ID = self.MRI_name[index]
        CT_preprocess = sitk.ReadImage(os.path.join(self.CT_path ,CT_ID),sitk.sitkFloat32)
        MRI_preprocess = sitk.ReadImage(os.path.join(self.MRI_path,MRI_ID),sitk.sitkFloat32)
        # Load input and target
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
        CT_preprocess = sitk.GetArrayFromImage(CT_preprocess)
        
        MRI_preprocess = sitk.GetArrayFromImage(MRI_preprocess)
        CT_preprocess = torch.FloatTensor(CT_preprocess)
        MRI_preprocess = torch.FloatTensor(MRI_preprocess)
        MRI_preprocess = MRI_preprocess.unsqueeze(0)
        MRI_preprocess1 = to_one_hot_3d(MRI_preprocess,2)
        print("CT:",CT_preprocess.shape)
        print("MRI:",MRI_preprocess.shape)
        print("Onehot:",MRI_preprocess1.shape)
                                   
            
        return CT_ID, MRI_ID, CT_preprocess, MRI_preprocess, MRI_preprocess1
            
    def __len__(self):
        #----------------------------------------
        # Indicate the total size of the dataset
        #----------------------------------------
        return len(self.CT_name)



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = F.interpolate(g1, scale_factor = (x1.shape[2]/g1.shape[2],0.5,0.5), mode = 'trilinear')
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor =2),
            nn.Conv3d(ch_in,ch_out,3,stride =1 ,padding =1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inpalce =True)
        )
    def forward(self, x):
        x =self.up(x)
        return x    



class Att_Unet(nn.Module):
    def __init__(self, in_channel = 4, out_channel = 2, training = True):
        super(Att_Unet, self).__init__()
        self.trianing = training
        # encoder section
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, 2, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace = True),
            nn.Conv3d(2, 4, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace = True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(4, 4, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace = True),
            nn.Conv3d(4, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(8, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(16, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.att1 = Attention_block(F_g =16, F_l =16, F_int =16)
        self.encoder5 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )

        # decoder section
        self.decoder1 = nn.Sequential(
            nn.Conv3d(48, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.att2 = Attention_block(F_g=8, F_l=16, F_int =16)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(16, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(32, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True)
        )
        self.att3 = Attention_block(F_g =4, F_l=8, F_int =8)
        self.decoder4 = nn.Sequential(
            nn.Conv3d(8, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv3d(16, 4, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace = True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv3d(4, out_channel, 3, stride = 1, padding = 1)
        )


        
    def forward(self, x):
        # encoder section
        #print('origin:',x.shape)
        x = self.encoder1(x)# relu(4->8)
        f1 = x #(8)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(8->8)
        x = self.encoder2(x)# relu(8->16)
        f2 = x #(16)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(16->16)
        x = self.encoder3(x)# relu(16->32)
        f3 = x #(32)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(32->32)
        x = self.encoder4(x)# relu(32->32)
        a1 = x
        A1 = self.att1(g = f3, x =a1)#attention block1
        #print('A1:',A1.shape)
        x = self.encoder5(x)#relu(32->64)
        
        # decoder section

        x = F.interpolate(x, scale_factor = (f3.shape[2]/x.shape[2],2,2), mode = 'trilinear')# upsample（16->16)
        #print('x:',x.shape)
        A1 = F.interpolate(A1, scale_factor = (f3.shape[2]/A1.shape[2],2,2), mode = 'trilinear')# upsample（16->16)
        #print('A1:',A1.shape)
        x = torch.cat((x,A1),1) #(64+32 = 96)
        #print('x:',x.shape)
        x = self.decoder1(x)# relu(96 ->32)
        #print('x:',x.shape)
        a2 = x#attention block2, size = 32
        #print('a2:',a2.shape)
        #print('f2:',f2.shape)
        A2 = self.att2(g =f2,x=a2) # size =32
        #print('A2',A2.shape)
        A2 = F.interpolate(A2, scale_factor = (f2.shape[2]/A2.shape[2],2,2), mode = 'trilinear')# upsample（16->16)

        #print("decoder1_size:",x.shape)
        x = self.decoder2(x)#relu(32->32)
        #print('x:',x.shape)
        x = F.interpolate(x, scale_factor =(f2.shape[2]/x.shape[2],2,2), mode = 'trilinear')# upsample(256->256)
        #print('A2:',A2.shape)
        x = torch.cat((x,A2),1) #(4+4 = 8) 
        #print('x:',x.shape)
        x = self.decoder3(x) # relu(8 ->2)
        #print('x:',x.shape)
        #print("decoder2_size:",x.shape)
        a3 = x#attention block2
        #print('a3:',a3.shape)
        A3 = self.att3(g =f1,x=a3)
        #print('x:',x.shape)
        A3 = F.interpolate(A3, scale_factor = (f1.shape[2]/A3.shape[2],2,2), mode = 'trilinear')# upsample（16->16)
        #print('A3:',A3.shape)
        x = self.decoder4(x)#relu(2->2)
        #print('x:',x.shape)
        x = F.interpolate(x, scale_factor =(f1.shape[2]/x.shape[2],2,2), mode = 'trilinear')# upsample(128->128)
        #print('x:',x.shape)
        x = torch.cat((x,A3),1) #(2+2 = 4)
        #print('x:',x.shape) 
        x = self.decoder5(x) # relu(4 ->2)
        #print('x:',x.shape)
        #print("decoder3_size:",x.shape)
        x = self.decoder6(x)
        #print('x:',x.shape)
        
        return x   
