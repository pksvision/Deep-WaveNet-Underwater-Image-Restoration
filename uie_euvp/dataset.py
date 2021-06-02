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
import glob


class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['hazy'], sample['clean']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        return {'hazy': hazy_image,
                'clean': clean_image}


class Dataset_Load(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.filesA, self.filesB = self.get_file_paths(self.data_path, 'EUVP')
        self.len = min(len(self.filesA), len(self.filesB))
        self.transform = transform
      
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hazy_im = cv2.resize(cv2.imread(self.filesA[index % self.len]), (256,256),
                                 interpolation=cv2.INTER_AREA)

        hazy_im = hazy_im[:, :, ::-1] ## BGR to RGB   
        hazy_im = np.float32(hazy_im) / 255.0


        clean_im = cv2.resize(cv2.imread(self.filesB[index % self.len]), (256,256),
                                  interpolation=cv2.INTER_AREA)

        clean_im = clean_im[:, :, ::-1] ## BGR to RGB   
        clean_im = np.float32(clean_im) / 255.0

        sample = {'hazy': hazy_im, 
                  'clean': clean_im}    
        if self.transform != None:
            sample = self.transform(sample)
    
        return sample


    def get_file_paths(self, root, dataset_name):
        if dataset_name=='EUVP':
            filesA, filesB = [], []
            sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
            for sd in sub_dirs:
                filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
                filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
        elif dataset_name=='UFO-120':
                filesA = sorted(glob.glob(os.path.join(root, 'lrd') + "/*.*"))
                filesB = sorted(glob.glob(os.path.join(root, 'hr') + "/*.*"))
        return filesA, filesB 
