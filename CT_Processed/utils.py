'''
Author: Feng-Chiao, Lee
Date: 2022-03
Edited:2023-01-16

CT Preprocessing Fucntion Code
'''




import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import dicom2nifti
import nibabel as nib
import scipy
import sys
from sklearn import preprocessing
from skimage import exposure
import os
import pandas as pd
from scipy.ndimage import label
from scipy.ndimage.morphology import generate_binary_structure
import SimpleITK as sitk
from pathlib import Path
import skimage.morphology as sm
import glob
from shutil import copy
import pandas as pd
import itk
import itkwidgets


class Utils:
    """
    The utils will include some funtion we will use in for Preprocessing
    
    """

    def sitk2itk(sitk_img, IMAGE_DIMENSION):
        """
        sitk_img: the input SimpleITK format image
        IMAGE_DIMENSION: the image dimension of sitk_img
        sitk2itk help us transfer the Simpleitk Image to itk Image; 
        we will inherit the origin and spacing from Simpleitk Image     
        """
        npsitk_img = sitk.GetArrayFromImage(sitk_img)
        itk_img = itk.GetImageFromArray(npsitk_img)
        itk_img.SetOrigin(sitk_img.GetOrigin())
        itk_img.SetSpacing(sitk_img.GetSpacing())
        itk_img.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_img.GetDirection()), [IMAGE_DIMENSION]*2)))
        return itk_img
    def itk2sitk(itk_img):
        """
        input: itk_img
        output: sitk_img
        itk2sitk help us transfer the itk Image to Simpleitk Image; 
        we will inherit the origin and spacing from itk Image     
        """
        npitk_img = itk.GetArrayFromImage(itk_img)
        sitk_img = sitk.GetImageFromArray(npitk_img, isVector = itk_img.GetNumberOfComponentsPerPixel()>1)
        sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
        sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
        sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
        return sitk_img

    
    def adjust_array(ori_direction, ori_array): 
        """
        adjust_array function help us flip our image into same direction
        ori_direction: the input image direction
        ori_array: the input image transfer to image 
        """
        ori_array = sitk.GetArrayFromImage(ori_array)
        print(ori_array.shape)
        if ori_direction[0] == -1.0:
            print('flip on axis 2')
            np.flip(ori_array, axis = 2)
        if ori_direction[4] == -1.0:
            np.flip(ori_array, axis = 1)
            print('flip on axis 1')
        if ori_direction[8] == -1.0:
            np.flip(ori_array, axis = 0)
            print('flip on axis 0')
            
        transfer_array = sitk.GetImageFromArray(ori_array)
        return transfer_array

    def CheckMiss(img_path, thres_prob):
        """
        img_path: the input image you will like to check
        thres_prob: slice difference partical between slice

        This is a function we use to check the completed of image;
        Output true/false of normal/abnormal image 
        """
        testtar_img = sitk.ReadImage(img_path)
        testtar_img = sitk.GetArrayFromImage(testtar_img)
        testtar_img[testtar_img>0] =1
        testtar_img[testtar_img<=0] =0
        numb = []
        for i in range(testtar_img.shape[0]):
            numb.append(sum(sum(testtar_img[i,:,:])))
        for i in range(len(numb)-1):
            short = i
            fast = i+1
            if ((float(abs(numb[i+1]-numb[i]))/numb[i]))>thres_prob:
                return False
                break
        return True

    def imageCopy(mask_dir,filtered_dir,goal_dir):
        """
        Copy Data Block
        Copy the origin, mask nifti file in the folder name CT_Defaced_finish
        """
        for count, name in enumerate(os.path.join(filtered_dir,"*")):
            pat_name = os.path.split(name)[-1]
            print(f'============{count+1}=============')
            save_path = os.path.join(goal_dir,os.path.split(name)[-1])
            print(save_path)
            os.makedirs(save_path,exist_ok = True)
            os.chdir(save_path)
            mask_path = os.path.join(mask_dir,pat_name)
            filtered_path = os.path.join(filtered_dir,pat_name)
            for root, dirs,files in os.walk(mask_path):
                for file in files:
                    copyfile = os.path.join(root,file)
                    copy(copyfile,save_path)
            for root, dirs,files in os.walk(filtered_path):
                for file in files:
                    copyfile = os.path.join(root,file)
                    copy(copyfile,save_path)
            print(os.listdir(save_path))
            print('==============copied==============')







