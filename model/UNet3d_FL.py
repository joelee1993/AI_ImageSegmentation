
"""
Author: Joe
Date: 07-22-2022
Edited: 01-28-2023
Try the basic 3dUnet code on stroke dataset with higher feature layer;
Extract a Simple Version from previous file

Basic 3dUnet code for training testing brain tumour dataset

"""
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

#from Dataset import CTDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
#from Unet_3d_model import Unet_3d
import numpy as np
import SimpleITK as sitk
import torchvision 
import torch.utils.data as data
import os
#import Common
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import tensorboard

def get_args( args : list) -> dict:

    parser = argparse.ArgumentParser(description ='3dUnet command line argument parser')
    parser.add_argument('--mode',
                        help = 'the action you want to do [train],[test]',
                        type = str,
                        choices =["train", "predict"],
                        required = True)
    parser.add_argument('--train_data_dir',
                        help = 'directory contains training data',
                        type = str)
    parser.add_argument('--label_data_dir',
                        help = 'directory contains label data',
                        type = str)
    parser.add_argument('--model_save_dir',
                        help = 'directory to save the model checkpoint',
                        type = str)
    parser.add_argument('--lr',
                        help = 'learning rate',
                        type = float)
    parser.add_argument('--epochs',
                        help = 'Epochs for training',
                        type = int)
    parser.add_argument('--bs',
                        help = 'batch size',
                        type = int)
    parser.add_argument('--model_save_name',
                        help = 'the training result model name',
                        type = str)
    parser.add_argument('--tensorboard_save_dir', 
                        help = 'The directory to save the tensorbroad data',
                        type = str)
    options = vars(parser.parse_args())
    return options




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
        print(MRI_ID)
        CT_preprocess = sitk.ReadImage(os.path.join(self.CT_path ,CT_ID),sitk.sitkFloat32)
        MRI_preprocess = sitk.ReadImage(os.path.join(self.MRI_path,MRI_ID),sitk.sitkFloat32)
        # Load input and target
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
        CT_preprocess = sitk.GetArrayFromImage(CT_preprocess)        
        MRI_preprocess = sitk.GetArrayFromImage(MRI_preprocess)
        CT_preprocess = torch.LongTensor(CT_preprocess)
        MRI_preprocess = torch.LongTensor(MRI_preprocess)
        CT_preprocess = CT_preprocess.unsqueeze(0)
        MRI_preprocess = MRI_preprocess.unsqueeze(0)

                                   
            
        return CT_ID, MRI_ID, CT_preprocess, MRI_preprocess
            
    def __len__(self):
        #----------------------------------------
        # Indicate the total size of the dataset
        #----------------------------------------
        return len(self.CT_name)




    
class Unet_3d(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, training = True):
        super(Unet_3d, self).__init__()
        self.trianing = training
        # encoder section
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, 4, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace = True),
            nn.Conv3d(4, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(8, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(16, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        )

        # decoder section
        self.decoder1 = nn.Sequential(
            nn.Conv3d(96, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(48, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(24, 8, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 4, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace = True),
            nn.Conv3d(4, 1, 1)
        )

        
    def forward(self, x):
        # encoder section
        x = self.encoder1(x)# relu(1->8)
        f1 = x #(8)
        print("encoder1_size:",f1.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(64->64)
        print("test",x.shape)
        x = self.encoder2(x)# relu(8->16)
        f2 = x #(8)
        print("encoder2_size:",f2.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(128->128)
        x = self.encoder3(x)# relu(16->32)
        f3 = x #(8)
        print("encoder3_size:",f3.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(256->256)
        x = self.encoder4(x)# relu(32->64)
        print("endcoder4_size:", x.shape)
        
        # decoder section

        x = F.interpolate(x, scale_factor = (f3.shape[2]/x.shape[2],f3.shape[3]/x.shape[3],f3.shape[4]/x.shape[4]), mode = 'trilinear')# upsample(512->512)
        print("bottleneck_size:",x.shape)
        x = torch.cat((x,f3),1) #(64+32 = 96)
        print("cat x: ", x.shape)
        x = self.decoder1(x)# relu(96 ->32)
        print("decoder1_size:",x.shape)
        x = F.interpolate(x, scale_factor =(f2.shape[2]/x.shape[2],f2.shape[3]/x.shape[3],f2.shape[4]/x.shape[4]), mode = 'trilinear')# upsample(256->256)
        x = torch.cat((x,f2),1) #(32+16 = 48)
        print("cat x: ", x.shape) 
        x = self.decoder2(x) # relu(48 ->16)
        print("decoder2_size:",x.shape)
        x = F.interpolate(x, scale_factor =(f1.shape[2]/x.shape[2],f1.shape[3]/x.shape[3],f1.shape[4]/x.shape[4]), mode = 'trilinear')# upsample(128->128)
        x = torch.cat((x,f1),1) #(8+16 = 24) 
        print("cat x: ", x.shape)
        x = self.decoder3(x) # relu(24 ->1)
        print("decoder3_size:",x.shape)
        x = F.interpolate(x, scale_factor =(f1.shape[2]/x.shape[2],f1.shape[3]/x.shape[3],f1.shape[4]/x.shape[4]), mode = 'trilinear')# upsample(128->128)
        # m = nn.Sigmoid()
        # x = m(x)
        return x       


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, smooth =1e-5 ):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = target.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum()+targets.sum()+smooth)
        return 1.0 - dice

 """
 Following Loss Function is the source code from
 https://github.com/wolny/pytorch-3dunet
 """       
def flatten(tensor):
      C = tensor.size(1)
      axis_order = (1, 0) + tuple(range(2, tensor.dim()))
      transposed = tensor.permute(axis_order)
      return transposed.contiguous().view(C,-1)
from torch.autograd import Variable


class WeigthedCrossEntropyLoss(nn.Module):
      def __init__(self, ignore_index =-1):
            super(WeigthedCrossEntropyLoss,self).__init__()
            self.ignore_index = ignore_index
      def forward(self, input, target):
            #weight = self._class_weights(input)
            input = input.squeeze(1)
            target = target.squeeze(1)
            input = input.long()
            target = target.long()
            return F.cross_entropy(input, target, weight = None, ignore_index = self.ignore_index)

      def _class_weights(input):
            input = F.softmax(input, dim =1)
            flattened = flatten(input)
            nominator = (1. - flattened).sum(-1)
            denominator = flattened.sum(-1)
            class_weights = Variable(nominator/denominator, requires_grad = False)
            return class_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma =2, weight = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        B,C,D,H,W = input.size()  
        input = input.view(B,C,D,-1)
        target = target.view(B,D,-1)
        logpt = F.log_softmax(input, dim =1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma*logpt
        logpt = logpt.long()
        loss = F.nll_loss(logpt,target, self.weight)
        return loss




def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)




from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import SimpleITK as sitk
import torchvision 
import torch.utils.data as data
# from loss import DiceLoss
# from Common import print_network
# from cus_argparse import get_args
import sys
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
device  = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
print(device)



def train(model, model_name, optimizer, loss_metric, lr, epochs, train_dataloader, val_dataloader, test_dataloader, tensor_save,**kwargs):
    net = model.to(device)
    if torch.cuda.device_count()>1:
          print('Lets use',torch.cuda.device_count(),'GPU!')
    optimizer = optimizer(net.parameters(), lr = lr)
    writer = SummaryWriter(tensor_save)
    #loss_metric = loss_fcn
    val_loss_old = 10000
    val_loss_unchanged_counter = 0
    #Common.print_network(net)# keep modify
    train_loss_list = []
    val_loss_list = []
    train_dice_list =[]
    val_dice_list = []
    train_name_list =[]
    val_name_list =[]
    test_name_list =[]

    print("Start Training")
    for epoch in tqdm(range(epochs), total = epochs):
        print(f"==========Epoch:{epoch+1}==========lr:{lr}========{epoch+1}/{epochs}")
        loss_sum = 0
        dice_sum = 0
        train_batch_count = 0
        val_batch_count = 0
        for index, (ctid, mriid, img, mask)  in enumerate(train_dataloader):
            img, mask = img.float(), mask.float()
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            output = net.forward(img)
            loss = loss_metric(output, mask)
            dice = 1-loss
            loss.backward()
            loss_sum += loss
            dice_sum += dice
            optimizer.step()
            train_name_list.append(ctid)
            train_batch_count = train_batch_count+1
        train_loss = loss_sum.item()/train_batch_count
        train_dice = dice_sum.item()/train_batch_count
        writer.add_scalar("train loss",float(train_loss),epoch)
        writer.add_scalar("train dice",float(train_dice),epoch)
        train_loss_list.append(train_loss)
        train_dice_list.append(train_dice)
            
        with torch.no_grad():
            loss_sum = 0
            dice_sum = 0
            for index,(ctid, mriid, img, mask) in enumerate(val_dataloader):
                img, mask = img.float(), mask.float()
                img, mask = img.to(device), mask.to(device)
                output = net.forward(img)
                loss = loss_metric(output, mask)
                dice = 1-loss
                loss_sum += loss
                dice_sum += dice 
                val_name_list.append(ctid)
                val_batch_count = val_batch_count+1
            #print(val_batch_count)
            val_loss = loss_sum.item()/val_batch_count
            val_dice = dice_sum.item()/val_batch_count
            writer.add_scalar("valid loss",float(val_loss),epoch)
            writer.add_scalar("valid dice",float(val_dice),epoch)
            val_loss_list.append(val_loss)
            val_dice_list.append(val_dice)
            for index,(ctid, mriid, img, mask) in enumerate(test_dataloader):
                test_name_list.append(ctid)
            print('Epoch:',epoch+1)
            print('-'*20)
            print('Train Loss:',train_loss)
            print('Train Dice Score:',train_dice)
            print('-'*20)
            print('Validation Loss:',val_loss)
            print('Validation Dice Score:',val_dice)
            print('-'*20)
            torch.save({
            'epoch': epoch+1,
            'model': model,
            'model_state_dict': net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice_list,
            'train_dice': train_dice_list,
            'val_loss': val_loss_list,
            'train_loss': train_loss_list,
            'train_name': train_name_list,
            'val_name':val_name_list,
            'test_name': test_name_list
            },os.path.join(MODEL_SAVE_DIR,f"{BESTMODEL_NAME}.pt"))
            torch.save(net.module.state_dict(),os.path.join(MODEL_SAVE_DIR,f"{BESTMODEL_NAME}_state.pt"))
            print("model saved")

if __name__ == '__main__':
    arguments = get_args(sys.argv)
    TENSOR = arguments.get('tensorboard_save_dir')
    MODE = arguments.get('mode')
    TRAIN_DIR = arguments.get('train_data_dir')
    LABEL_DIR = arguments.get('label_data_dir')

    if MODE == 'train':
        BESTMODEL_NAME = arguments.get('model_save_name')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        EPOCHS = arguments.get('epochs')
        BATCH_SIZE = arguments.get('bs')

        CT_path = TRAIN_DIR
        MRI_path = LABEL_DIR
        train_set = CTDataset(CT_image_root = CT_path, MRI_label_root = MRI_path)
        train_set_size = int(train_set.__len__()*0.8)
        test_set_size = len(train_set) - train_set_size
        train_set, test_set = data.random_split(train_set, [train_set_size, test_set_size])
        train_set_size1 = int(len(train_set)*0.9)
        valid_set_size = train_set_size - train_set_size1
        train_set, valid_set = data.random_split(train_set, [train_set_size1, valid_set_size])
        print("Train data set:",len(train_set))
        print("Valid data set:", len(valid_set))
        print("Test data set:", len(test_set))
        


        Batch_Size = BATCH_SIZE
        train_loader = DataLoader(dataset = train_set, batch_size = Batch_Size)
        valid_loader = DataLoader(dataset = valid_set, batch_size = Batch_Size)
        test_loader = DataLoader(dataset = test_set, batch_size = Batch_Size)


        model = Unet_3d()
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3])
        model = model.to(device)
        model_name = "Unet_3d"
        loss_fcn = WeigthedCrossEntropyLoss()
        optimizer = optim.Adam

        output = train(model, 
                    model_name, 
                    optimizer, 
                    loss_metric = loss_fcn, 
                    lr =LEARNING_RATE, 
                    epochs = EPOCHS, 
                    train_dataloader = train_loader,
                    val_dataloader = valid_loader,
                    test_dataloader = test_loader,
                    tensor_save = TENSOR)

    print('Train Finished')

