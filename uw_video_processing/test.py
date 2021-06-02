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
# from options import opt
import math
import shutil
from tqdm import tqdm

CHECKPOINTS_DIR = './ckpts'
# INP_DIR = opt.testing_dir_inp
# CLEAN_DIR = opt.testing_dir_gt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        

ch = 3

network = CC_Module()
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_295.pt"))
network.load_state_dict(checkpoint['model_state_dict'])
network.eval()
network.to(device)

# result_dir = './facades/video/enhanced_frames/'
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

def frames_to_video(noisy_video, clean_video, save_path, fps):
   image_array = []
   for i in range(len(noisy_video)):
       
       noisy_img = noisy_video[i]
       clean_img = clean_video[i]

       h_img = cv2.hconcat([noisy_img, clean_img])
       size =  (h_img.shape[1], h_img.shape[0])

       image_array.append(h_img)
       # cv2.imwrite('./new_frames/'+str(i)+'.jpg', h_img)
       # print(i)

   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(save_path,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])

   # cv2.destroyAllWindows()
   out.release()
   print('Done')

def get_frames(video_path):
    vid_frames = []
    vid = cv2.VideoCapture(video_path)
    #for frame identity
    # index = 0
    while(True):
        # Extract images
        ret, frame = vid.read()
        # end of frames
        if not ret: 
            break
        # Saves images
        # name = './test_frames/'+ str(index) + '.png'
        # print ('Creating...' + name)
        # cv2.imwrite(name, frame)
        vid_frames.append(frame)

        # next frame
        # index += 1
    return vid_frames

if __name__ =='__main__':

    file_name = 'degraded_video.mp4'
    total_files = get_frames(file_name)
    output_frames = []
    # total_files = os.listdir('./test_frames/')
    st = time.time()
    with tqdm(total=len(total_files)) as t:

        for m in range(len(total_files)):
        
            img = total_files[m]
            img = img[:, :, ::-1]   
            img = np.float32(img) / 255.0
            h,w,c=img.shape

            train_x = np.zeros((1, ch, h, w)).astype(np.float32)

            train_x[0,0,:,:] = img[:,:,0]
            train_x[0,1,:,:] = img[:,:,1]
            train_x[0,2,:,:] = img[:,:,2]
            dataset_torchx = torch.from_numpy(train_x)
            dataset_torchx=dataset_torchx.to(device)

            output=network(dataset_torchx)
            output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = output[:, :, ::-1]
            # cv2.imwrite(os.path.join(result_dir + str(total_files[m])), output)
            output_frames.append(output)

            t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
            t.update(1)
            
    end = time.time()
    print('Total time taken in secs : '+str(end-st))
    print('Per image (avg): '+ str(float((end-st)/len(total_files))))

    outpath =  './enhanced_video.avi'
    fps = 29
    frames_to_video(total_files, output_frames, outpath, fps)