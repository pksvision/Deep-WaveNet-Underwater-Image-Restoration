from __future__ import absolute_import, division, print_function

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import torchvision.models as pt_models
import dataset as dataset
from vgg import *

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from options import opt, device
from models import *
from misc import *
# from progress.bar import Bar
import re
import sys
from ssim import *


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


if __name__ == '__main__':

	scale = opt.scale
	print("Underwater Image Super-Resolution [SCALE]: ", scale)

	netG = CC_Module(scale)
	# print('underwater dehaze network ', netG)
	netG.to(device)

	mse_loss = nn.MSELoss()
	ssim_loss = SSIMLoss(11)

	vgg = Vgg16(requires_grad=False).to(device)

	optim_g = optim.Adam(netG.parameters(), 
						 lr=opt.learning_rate_g, 
						 betas = (opt.beta1, opt.beta2), 
						 weight_decay=opt.wd_g)

		
	dataset = dataset.Dataset_Load(data_path = opt.data_path,
								   scale=scale,
								   transform=dataset.ToTensor()
								   )
	batches = int(dataset.len / opt.batch_size)

	dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
	
	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)
	
	models_loaded = getLatestCheckpointName()    
	latest_checkpoint_G = models_loaded
	
	print('loading model for generator ', latest_checkpoint_G)
	
	if latest_checkpoint_G == None :
		start_epoch = 1
		print('No checkpoints found for netG and netD! retraining')
	
	else:
		checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))    
		start_epoch = checkpoint_g['epoch'] + 1
		netG.load_state_dict(checkpoint_g['model_state_dict'])
		optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
		for param_group in optim_g.param_groups:
			param_group['lr'] = opt.learning_rate_g
			
		print('Restoring model from checkpoint ' + str(start_epoch))
	
	netG.train()


	for epoch in range(start_epoch, opt.end_epoch + 1):
		
		# bar = Bar('Training', max=batches)
	
		opt.total_mse_loss = 0.0
		opt.total_vgg_loss = 0.0
		opt.total_ssim_loss = 0.0
		opt.total_G_loss = 0.0
			
		for i_batch, sample_batched in enumerate(dataloader):

			hazy_batch = sample_batched['hazy']
			clean_batch = sample_batched['clean']

			hazy_batch = hazy_batch.to(device)
			clean_batch = clean_batch.to(device)

			optim_g.zero_grad()

			
			pred_batch = netG(hazy_batch)
			batch_mse_loss = torch.mul(opt.lambda_mse, mse_loss(pred_batch, clean_batch))
			batch_mse_loss.backward(retain_graph=True)
			
			batch_ssim_loss = torch.mul(opt.lambda_ssim, ssim_loss(pred_batch, clean_batch))
			batch_ssim_loss.backward(retain_graph=True)

			clean_vgg_feats = vgg(normalize_batch(clean_batch))
			pred_vgg_feats = vgg(normalize_batch(pred_batch))
			batch_vgg_loss = torch.mul(opt.lambda_vgg, mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2))
			batch_vgg_loss.backward()
			
			opt.batch_mse_loss = batch_mse_loss.item()
			opt.total_mse_loss += opt.batch_mse_loss

			opt.batch_ssim_loss = batch_ssim_loss.item()
			opt.total_ssim_loss += opt.batch_ssim_loss

			opt.batch_vgg_loss = batch_vgg_loss.item()
			opt.total_vgg_loss += opt.batch_vgg_loss
			
			opt.batch_G_loss = opt.batch_mse_loss + opt.batch_vgg_loss + opt.batch_ssim_loss
			opt.total_G_loss += opt.batch_G_loss
			
			optim_g.step()

			# bar.suffix = f' Epoch : {epoch} | ({i_batch+1}/{batches}) | ETA: {bar.eta_td} | g_mse: {opt.batch_mse_loss} | g_vgg: {opt.batch_vgg_loss}'
			print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | mse: ' + str(opt.batch_mse_loss) + ' | vgg: ' + str(opt.batch_vgg_loss) + ' | ssim: ' + str(opt.batch_ssim_loss), end='', flush=True)
			# bar.next()
			
 

		print('\nFinished ep. %d, lr = %.6f, total_mse = %.6f, total_vgg = %.6f, total_ssim = %.6f' % (epoch, get_lr(optim_g), opt.total_mse_loss, opt.total_vgg_loss, opt.total_ssim_loss))
			# print('training epoch %d, %d / %d patches are finished, g_mse = %.6f' % (
			 # epoch, i_batch, batches, opt.batch_mse_loss))

		torch.save({'epoch':epoch, 
					'model_state_dict':netG.state_dict(), 
					'optimizer_state_dict':optim_g.state_dict(), 
					'mse_loss':opt.total_mse_loss, 
					'vgg_loss':opt.total_vgg_loss, 
					'ssim_loss':opt.total_ssim_loss, 
					'opt':opt,
					'total_loss':opt.total_G_loss}, os.path.join(opt.checkpoints_dir, 'netG_' + str(epoch) + '.pt'))