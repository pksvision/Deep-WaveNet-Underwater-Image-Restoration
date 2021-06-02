from models import CC_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from options import opt
import math
from measure_ssim_psnr import *
import shutil
from tqdm import tqdm
from get_uiqm import *

CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        

ch = 3
SCALE = opt.scale
print("Resolution scale : ", SCALE)

atmos_net = CC_Module(SCALE)
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_1603.pt"))
atmos_net.load_state_dict(checkpoint['model_state_dict'])
atmos_net.eval()
atmos_net.to(device)

if (SCALE==2): lr_res_ = (320, 240)
elif (SCALE==3): lr_res_ = (214, 160)
else: lr_res_ = (160, 120)

result_dir = './facades/Ours_X'+str(SCALE)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if __name__ =='__main__':

    total_files = os.listdir(INP_DIR)
    st = time.time()
    with tqdm(total=len(total_files)) as t:

        for m in total_files:
        
            img=cv2.resize(cv2.imread(INP_DIR + str(m)), lr_res_, cv2.INTER_CUBIC)
            img = img[:, :, ::-1]   
            img = np.float32(img) / 255.0
            h,w,c=img.shape

            train_x = np.zeros((1, ch, h, w)).astype(np.float32)

            train_x[0,0,:,:] = img[:,:,0]
            train_x[0,1,:,:] = img[:,:,1]
            train_x[0,2,:,:] = img[:,:,2]
            dataset_torchx = torch.from_numpy(train_x)
            dataset_torchx=dataset_torchx.to(device)

            output=atmos_net(dataset_torchx)
            output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = output[:, :, ::-1]
            cv2.imwrite(os.path.join(result_dir + str(m)), output)

            t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
            t.update(1)
            
    end = time.time()
    print('Total time taken in secs : '+str(end-st))
    print('Per image (avg): '+ str(float((end-st)/len(total_files))))

    ### compute SSIM and PSNR
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(CLEAN_DIR, result_dir)
    print("SSIM on {0} samples".format(len(SSIM_measures))+"\n")
    print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures))+"\n")
    print("PSNR on {0} samples".format(len(PSNR_measures))+"\n")
    print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures))+"\n")
    measure_UIQMs(result_dir)
    # shutil.rmtree(result_dir)