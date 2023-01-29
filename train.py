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
from utils import to_one_hot_3d
from AttUnetModel import CTDataset
from AttUnetModel import Attention_block
from AttUnetModel import Att_Unet
import losses
from utils import BalancedDataParallel
def get_args( args : list) -> dict:

    parser = argparse.ArgumentParser(description ='3dUnet command line argument parser')
    parser.add_argument('--mode',
                        help = 'the action you want to do [train],[test]',
                        type = str,
                        choices =["train", "predict"],
                        required = True)

    parser.add_argument('--model_save_name',
                        help = 'the training result model name',
                        type = str)

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
    parser.add_argument('--lossfcn',
                        help = 'choose the loss function',
                        type = str,
                        choices = ["BCEDiceLoss", "DiceLoss"],
                        default = 'DiceLoss')
    options = vars(parser.parse_args())
    return options




def train(model, model_name, optimizer, loss_metric, lr, epochs, train_dataloader, val_dataloader, test_dataloader, **kwargs):
    net = model
    if torch.cuda.device_count()>1:
        print('Lets use',torch.cuda.device_count(),'GPU!')
    optimizer = optimizer(net.parameters(), lr = lr)
    val_loss_old = 10000
    val_loss_unchanged_counter = 0
    train_loss_list = []
    val_loss_list = []
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
        for index, (ctid, mriid, img, mask, mask_onehot)  in enumerate(train_dataloader):
            img, mask_onehot = img.float(), mask_onehot.float()
            img, mask_onehot = img.to(device), mask_onehot.to(device)
            optimizer.zero_grad()
            output = net.forward(img)
            print(output.shape)
            print(mask_onehot.shape)
            loss = loss_metric(output, mask_onehot)
            loss.backward()
            loss_sum += loss
            optimizer.step()
            train_name_list.append(ctid)
            train_batch_count = train_batch_count+1
        train_loss = loss_sum.item()/train_batch_count
        train_loss_list.append(train_loss)
            
        with torch.no_grad():
            loss_sum = 0
            dice_sum = 0
            for index,(ctid, mriid, img, mask,mask_onehot) in enumerate(val_dataloader):
                img, mask_onehot = img.float(), mask_onehot.float()
                img, mask_onehot = img.to(device), mask_onehot.to(device)
                output = net.forward(img)
                loss = loss_metric(output, mask_onehot)
                loss_sum += loss
                val_name_list.append(ctid)
                val_batch_count = val_batch_count+1
            #print(val_batch_count)
            val_loss = loss_sum.item()/val_batch_count
            #val_dice = dice_sum.item()/val_batch_count
            val_loss_list.append(val_loss)
            #val_dice_list.append(val_dice)
            for index,(ctid, mriid, img, mask, mask_onehot) in enumerate(test_dataloader):
                test_name_list.append(ctid)
            print('Epoch:',epoch+1)
            print('-'*20)
            print('Train Loss:',train_loss)
            #print('Train Dice Score:',train_dice)
            print('-'*20)
            print('Validation Loss:',val_loss)
            #print('Validation Dice Score:',val_dice)
            print('-'*20)
            torch.save({
            'epoch': epoch+1,
            'model': model,
            'model_state_dict': net.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_list,
            'train_loss': train_loss_list,
            'train_name': train_name_list,
            'val_name':val_name_list,
            'test_name': test_name_list
            },os.path.join(MODEL_SAVE_DIR,f"{BESTMODEL_NAME}.pt"))
            print("model saved")

# if __name__ == '__main__':
arguments = get_args(sys.argv)
MODE = arguments.get('mode')
TRAIN_DIR = arguments.get('train_data_dir')
LABEL_DIR = arguments.get('label_data_dir')



if MODE == 'train':
    BESTMODEL_NAME = arguments.get('model_save_name')
    MODEL_SAVE_DIR = arguments.get('model_save_dir')
    LEARNING_RATE = arguments.get('lr')
    EPOCHS = arguments.get('epochs')
    BATCH_SIZE = arguments.get('bs')
    LOSS_FCN = arguments.get('lossfcn')
    print(LOSS_FCN)

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
    train_loader = DataLoader(dataset = train_set, batch_size = Batch_Size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_set, batch_size = Batch_Size, shuffle = True)
    test_loader = DataLoader(dataset = test_set, batch_size = Batch_Size, shuffle = True)
 
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    model = Att_Unet()
    model = model.to(device)
    gpu0_bsz = 1 
    acc_grad = 2
    model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).to(device)

    model_name = "AttUnet_Parallel_BCE"
    if LOSS_FCN == 'BCEDiceLoss':
      loss_fcn = losses.BCEDiceLoss(0.3, 0.7)
    elif LOSS_FCN == 'DiceLoss':
      loss_fcn = losses.DiceLoss
    
    optimizer = optim.Adam

    output = train(model, 
                   model_name, 
                   optimizer, 
                   loss_metric = loss_fcn, 
                   lr =LEARNING_RATE, 
                   epochs = EPOCHS, 
                   train_dataloader = train_loader,
                   val_dataloader = valid_loader,
                   test_dataloader = test_loader)

print('Train Finished')