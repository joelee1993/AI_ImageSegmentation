
'''
Author: Feng-Chiao, Lee
Date: 2022-03
Edited:2022-12-19

The code can help dealing with the raw CT data

'''

import cv2
import numpy as np
np.set_printoptions(threshold = np.inf)
import matplotlib.pyplot as plt
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
from utils import Utils


needs_skullstrip = []
for name in glob.glob(r'S:\StrokeAI\CT_Defaced_dataset\*'):
    pat_name = os.path.split(name)[-1]
    if os.path.isfile(os.path.join(name,f'{pat_name}_skullstrip.nii')):
        pass
    else:
        needs_skullstrip.append(pat_name)
needs_skullstrip = sorted(needs_skullstrip)        
print(len(needs_skullstrip))


class CT_Processed:

    def CT_preprocess_standard(nif_dat_path,template_path):
        """ Given an 3D nifti data, scale into window and level. Scale between
        expects a tuple (new_min, new_max) that determines the new range.The preprocess 
        also do the resampling and registration of the CT image.
        Works with both 2D and 3D data.
        
        nif_dat_path: input the image data path
        """
        
        MNI_152 = sitk.ReadImage(template_path) 
        nif_dat = sitk.ReadImage(nif_dat_path) 

        fixed_img = sitk.Cast(MNI_152, sitk.sitkFloat32)
        moving_img = sitk.Cast(nif_dat, sitk.sitkFloat32)
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img, # Note these are purposefuly reversed!
            moving_img,# Note these are purposefuly reversed!
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        moving_resampled = sitk.Resample(
            moving_img,       # Note these are purposefuly reversed!
            fixed_img,      # Note these are purposefuly reversed!
            initial_transform,
            sitk.sitkLinear,  # TODO: use different interpolator?
            0.0,              # Note(Jacob): default value
            moving_img.GetPixelID(),
        )
        npmoving = sitk.GetArrayFromImage(moving_resampled)
        #plt.figure(1)
        #plt.imshow(npmoving[80,:,:],cmap ='gray')
        #plt.colorbar()
        itk_fix = Utils.sitk2itk(fixed_img,3)
        itk_mov = Utils.sitk2itk(moving_resampled,3)
        #print(itk.size(itk_mov))
        parameter_object = itk.ParameterObject.New()
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(parameter_map_rigid)
        result_img, result_transform_parameters = itk.elastix_registration_method(itk_fix,itk_mov, parameter_object = parameter_object,log_to_console = True)
        result_img = Utils.itk2sitk(result_img)
        return result_img



    def CT_preprocess_final(nif_dat_path, new_spacing ,level, window, scale, fltsize, Origin):
        """ Given an 3D nifti data, scale into window and level. Scale between
        expects a tuple (new_min, new_max) that determines the new range.The preprocess 
        also do the resampling and registrationof the CT image.
        Works with both 2D and 3D data.
        
        nif_dat_path: input the data path of of nifti data
        new_spacing: the voxel size: [voxel[0], voxel[1], voxel[2]]
        level : center of window, window = width of window.
        scale : (min_scale, max_scale) to rescale in-window values to.
        filt_size : the median filter kernal size; Ex:fltsize = 5  = median_filter(5,5,5)
        Origin: The nifti data origin

        """
        
        MNI_152 = sitk.ReadImage(MNI_path) 
        nif_dat = sitk.ReadImage(nif_dat_path) 

        fixed_img = sitk.Cast(MNI_152, sitk.sitkFloat32)
        moving_img = sitk.Cast(nif_dat, sitk.sitkFloat32)
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img, # Note these are purposefuly reversed!
            moving_img,# Note these are purposefuly reversed!
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        moving_resampled = sitk.Resample(
            moving_img,       # Note these are purposefuly reversed!
            fixed_img,      # Note these are purposefuly reversed!
            initial_transform,
            sitk.sitkLinear,  # TODO: use different interpolator?
            0.0,              
            moving_img.GetPixelID(),
        )
        npmoving = sitk.GetArrayFromImage(moving_resampled)
        itk_fix = Utils.sitk2itk(fixed_img,3)
        itk_mov = Utils.sitk2itk(moving_resampled,3)
        parameter_object = itk.ParameterObject.New()
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(parameter_map_rigid)
        result_img, result_transform_parameters = itk.elastix_registration_method(itk_fix,itk_mov, parameter_object = parameter_object,log_to_console = True)
        result_img = Utils.itk2sitk(result_img)

        npmoving = sitk.GetArrayFromImage(result_img)
        npmoving = scipy.ndimage.median_filter(npmoving, size = (fltsize,fltsize,fltsize))    
        # Extract the brain
        im_size = npmoving.shape
        upper = level + window/2.0
        lower = level - window/2.0
        npmoving[npmoving > upper] = upper
        npmoving[npmoving < lower] = lower
        
        re_img = np.zeros(im_size)
        for num in range(im_size[0]):
            img1 = npmoving[num, :, :]
            new_img = img1.copy()        
            new_img[new_img > upper] = upper
            new_img[new_img < lower] = lower

        # Delete the Gantry
            if new_img.max() > 0: 
                if np.max(img1)<=1e-10:
                    low_t=0
                else:
                    low_t=1e-10
                ref, bi_img = cv2.threshold(img1 ,low_t,1,cv2.THRESH_BINARY)
                bi_img = np.uint8(bi_img)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bi_img)
                op = np.zeros(new_img.shape)
                pixel_label = np.resize(labels, labels.shape[0]*labels.shape[1])            
                num1 = np.delete(np.bincount(pixel_label),0)
                label_brain = np.where(num1 == num1.max())[0]+1
                mask = labels == label_brain
                brain_pos = np.where(mask[:,:] == True)
                op[mask] = new_img[brain_pos]
                op.astype(float)
                re_img[num, :, :] = op
            elif new_img.max() == 0:
                op = new_img
                re_img[num, :, :] = op
        re_img = (re_img -lower)/ (upper-lower) * (scale[1] - scale[0])
        re_img = sitk.GetImageFromArray(re_img)

        print('re_img size: ', re_img.GetSize())
        print('re_img spacing: ', re_img.GetSpacing())
    
        return re_img

    def coRegis_final(fixed_img, moving_img, moving_label,metric_method,ex_filterfile,CTMR_filterfile):
        """
        
        This code is help doing the co-registration between CT-image and MR Image/ MRI lesion label
        We do the registration first of the CT image and MR image. After that, we use the transformation
        matrix to transfer MR label to CT-label
        
        fixed_img: CT Image
        moving_img: MR Image
        moving_label: MR label
        metric_method:
        ["extremity MRI"]: intra-subject; affine transformation, mutual information metric
        ["CTMR-based"]:intra-subject; multi-resolution (4)-
        rigid + B-spline transformation, Mutual Information metric (Mattes) with Adaptive Stochastic Gradient Descent optimizer
        
        We tranfer MR image/ MR label into CT space by initial transform and do the co-registration,
        the MR image and MR label spacing/origin will change and same with CT image
        """
        # Generate a centering transform based on the images
        IMAGE_DIMENSION = 3
        # fixed_img = CT_preprocess_skulloff(fixed_img,40, 80, (0.0,1.0),3)
        fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)
        fixed_img = Utils.adjust_array(fixed_img.GetDirection(),fixed_img)
        processed_ct  = fixed_img
        fixed_flip = fixed_img


        moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
        moving_ori = moving_img.GetOrigin()
        moving_img = sitk.GetArrayFromImage(moving_img)
        moving_img  = sitk.GetImageFromArray(moving_img)
        moving_img.SetOrigin(moving_ori)
        moving_img = Utils.adjust_array(moving_img.GetDirection(),moving_img)

        moving_flip = moving_img
        print(moving_flip.GetDirection())

        label_img = sitk.Cast(moving_label, sitk.sitkFloat32)
        label_ori = label_img.GetOrigin()
        label_img = sitk.GetArrayFromImage(label_img)
        label_img  = sitk.GetImageFromArray(label_img)
        label_img.SetOrigin(label_ori)
        label_img = Utils.adjust_array(label_img.GetDirection(),label_img)
        label_flip = label_img
        print(label_flip.GetDirection())
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_flip, # Note these are purposefuly reversed!
            moving_flip,# Note these are purposefuly reversed!
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        
        moving_resampled = sitk.Resample(
            moving_flip,       # Note these are purposefuly reversed!
            fixed_flip,      # Note these are purposefuly reversed!
            initial_transform,
            sitk.sitkNearestNeighbor,# use different interpolator?
            0.0,              
            moving_flip.GetPixelID(),
        )
        label_resampled = sitk.Resample(
            label_flip,       # Note these are purposefuly reversed!
            fixed_flip,      # Note these are purposefuly reversed!
            initial_transform,
            sitk.sitkNearestNeighbor,# use different interpolator
            0.0,              
            label_flip.GetPixelID(),
        )
        print('Fixed size ', fixed_flip.GetSize())
        print('Moving size ', moving_flip.GetSize())
        print('Moving Resampled size', moving_resampled.GetSize())
        print('Label Resampled size', label_resampled.GetSize())

        print('Fixed space ', fixed_flip.GetSpacing())
        print('Moving space ', moving_flip.GetSpacing())
        print('Resampled space', moving_resampled.GetSpacing())
        print('Label Resampled space', label_resampled.GetSpacing())
        
        print('Fixed origin ', fixed_flip.GetOrigin())
        print('Moving origin ', moving_flip.GetOrigin())
        print('Resampled origin ', moving_resampled.GetOrigin())
        print('Label Resampled origin', label_resampled.GetOrigin())

        itk_fix = Utils.sitk2itk(fixed_flip,3)
        itk_mov = Utils.sitk2itk(moving_resampled,3)
        itk_label = Utils.sitk2itk(label_resampled,3)
        parameter_object1 = itk.ParameterObject.New()
        parameter_map_rigid1 = parameter_object1.GetDefaultParameterMap('rigid')
        parameter_map_rigid1['FinalBSplineInterpolationOrder'] = ['0']
        parameter_object1.AddParameterMap(parameter_map_rigid1)
        parameter_map_affine1 = parameter_object1.GetDefaultParameterMap('affine')
        parameter_map_affine1['FinalBSplineInterpolationOrder'] = ['0']
        parameter_object1.AddParameterMap(parameter_map_affine1)
        
        if metric_method == 'extremity MRI':
            parameter_object1.AddParameterFile(ex_filterfile)
        if metric_method == 'CTMR-based':
            parameter_object1.AddParameterFile(CTMR_filterfile)


        result_img, result_transform_parameters = itk.elastix_registration_method(itk_fix,itk_mov, parameter_object = parameter_object1,log_to_console = True)
        
        result_img = Utils.itk2sitk(result_img)
        itk_label = Utils.sitk2itk(label_resampled,3)       
        transform_label_img1 =itk.transformix_filter(itk_label, result_transform_parameters)
        label_result =Utils.itk2sitk(transform_label_img1)
        return label_result , result_img,label_resampled ,moving_resampled, result_transform_parameters, moving_flip, label_flip,processed_ct



