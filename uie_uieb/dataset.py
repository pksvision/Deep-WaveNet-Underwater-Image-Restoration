import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
from options import opt
import torchvision
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image



class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['hazy'], sample['clean']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        # hazy_image = hazy_image / 255.0
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        # clean_image = clean_image / 255.0
        return {'hazy': hazy_image,
                'clean': clean_image}


class Dataset_Load(Dataset):
    def __init__(self, hazy_path, clean_path, transform=None):
        self.hazy_dir = hazy_path
        self.clean_dir = clean_path
        self.transform = transform
      
    def __len__(self):
        return opt.num_images

    def __getitem__(self, index):
        hazy_image_name = str(index ) + opt.img_extension
        hazy_im = cv2.resize(cv2.imread(os.path.join(self.hazy_dir, hazy_image_name)), (512,512),
                                 interpolation=cv2.INTER_AREA)

        hazy_im = hazy_im[:, :, ::-1] ## BGR to RGB   
        hazy_im = np.float32(hazy_im) / 255.0


        clean_image_name = str(index ) + opt.img_extension
        clean_im = cv2.resize(cv2.imread(os.path.join(self.clean_dir, clean_image_name)), (512,512),
                                  interpolation=cv2.INTER_AREA)

        clean_im = clean_im[:, :, ::-1] ## BGR to RGB   
        clean_im = np.float32(clean_im) / 255.0

        sample = {'hazy': hazy_im, 
                  'clean': clean_im}    
        if self.transform != None:
            sample = self.transform(sample)
    
        return sample