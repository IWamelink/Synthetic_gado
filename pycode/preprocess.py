"""
@author: Ivar Wamelink

Load data and train model.
"""
import concurrent.futures
import time

import nibabel as nib
import numpy as np
import os

from glob import glob

# Inputs
t1 = sorted(glob('//home/iwamelink/projects/IMAGO/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/*/*t1.*'))
t2 = sorted(glob('//home/iwamelink/projects/IMAGO/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/*/*t2*'))
flair = sorted(glob('//home/iwamelink/projects/IMAGO/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/*/*flair*'))
seg = sorted(glob('//home/iwamelink/projects/IMAGO/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/*/*seg*'))

# Ground truth
t1ce = sorted(glob('//home/iwamelink/projects/IMAGO/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/*/*t1ce*'))

# Check if all sizes are equal
t1_im = nib.load(t1[0]).get_fdata()
t2_im = nib.load(t2[0]).get_fdata()
flair_im = nib.load(flair[0]).get_fdata()
seg_im = nib.load(seg[0]).get_fdata()
t1ce_im = nib.load(t1ce[0]).get_fdata()

if t1_im.shape != t2_im.shape and t1_im.shape != flair_im.shape and t1_im.shape != seg_im.shape and t1_im.shape != t1ce_im.shape:
    print('\033[31mshapes of the images are not constant')
else:
    del t1_im, t2_im, flair_im, seg_im, t1ce_im

def multi_norm_save(files):

    t1 = files[0]
    t2 = files[1]
    flair = files[2]
    t1ce = files[3]

    def norm_save(location):
        volume = nib.load(location)
        volume = volume.get_fdata()

        volume = (volume - np.nanmin(volume)) / (np.nanmax(volume) - np.nanmin(volume))

        return volume

    file = t1.split('/')[-2]

    header = nib.load(t1[i]).affine

    # Check if all files are from the same patient.
    if file == t2.split('/')[-2] and file == flair.split('/')[-2] and file == t1ce.split('/')[-2]:
        # Channel 0 is t1, channel 1 is t2, channel 2 is flair
        volume = np.zeros((240, 240, 155, 3))
        volume[..., 0] = norm_save(t1)
        volume[..., 1] = norm_save(t2)
        volume[..., 2] = norm_save(flair)
        img = nib.Nifti1Image(volume, header)
        img.to_filename('/home/iwamelink/projects/IMAGO/BraTS/normalized/input/{}.nii.gz'.format(file))

        volume = norm_save(t1ce)
        img = nib.Nifti1Image(volume, header)
        img.to_filename('/home/iwamelink/projects/IMAGO/BraTS/normalized/contrast/{}_t1ce.nii.gz'.format(file))
        del img, volume
    else:
        print('\033[31mFiles are not from the same patient:\033[38m ', file)

files = []
for i in range(len(t1)):
    files.append([t1[i], t2[i], flair[i], t1ce[i]])

with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
    executor.map(multi_norm_save, files)
