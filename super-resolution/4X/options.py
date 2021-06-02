import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./ufo_dataset/train_val')
parser.add_argument('--scale', type=int, default=4)

parser.add_argument('--checkpoints_dir', default='./ckpt/')
parser.add_argument('--batch_size', type=int, default=5)

parser.add_argument('--learning_rate_g', type=float, default=2e-04)

parser.add_argument('--end_epoch', type=int, default=1000)
parser.add_argument('--img_extension', default='.png')

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=0.02)
parser.add_argument('--lambda_ssim', type=float, default=0.5)


parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_dir_inp', default="../../Deep_SESR/lrd/")
parser.add_argument('--testing_dir_gt', default="../../Deep_SESR/hr/")

opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)