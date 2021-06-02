import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--hazydir', default='../K5/train/hazy_train/')
parser.add_argument('--cleandir', default='../K5/train/clean_train/')

parser.add_argument('--checkpoints_dir', default='./ckpts/')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_images', type=int, default=800)

parser.add_argument('--learning_rate_g', type=float, default=2e-04)

parser.add_argument('--end_epoch', type=int, default=1000)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=512)

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=0.02)


parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")
parser.add_argument('--testing_dir_inp', default="./hazy_test/")
parser.add_argument('--testing_dir_gt', default="./clean_test/")

opt = parser.parse_args()
# print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)