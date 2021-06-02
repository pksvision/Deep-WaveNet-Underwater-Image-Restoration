"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uiqm_utils import getUIQM
from tqdm import tqdm, trange
import sys

def measure_UIQMs(dir_name, im_res=(256, 256)):
	paths = sorted(glob(join(dir_name, "*.*")))
	uqims = []
	with tqdm(total=len(paths)) as t:
		t.set_description('Estimating Mean and Std-dev (MS) of UQIM for : {}'.format(dir_name))
		for img_path in range(len(paths)):
			im = Image.open(paths[img_path]).resize(im_res)
			uiqm = getUIQM(np.array(im))
			uqims.append(uiqm)
			t.set_postfix(MS='{:.6f} {:.6f}'.format(np.mean(uqims), np.std(uqims)))
			t.update(1)
	print (dir_name+" UIQMs >> Mean: {0} std: {1}".format(np.mean(uqims), np.std(uqims)))
	return np.array(uqims)

# """
# Get datasets from
#  - http://irvlab.cs.umn.edu/resources/euvp-dataset
#  - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
# """
# #inp_dir = "/home/xahid/datasets/released/EUVP/test_samples/Inp/"
# inp_dir = ".\\EP2320_Nat\\"
# ## UIQMs of the distorted input images 
# inp_uqims = measure_UIQMs(inp_dir)
# print (inp_dir+" UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

if __name__ == '__main__':
	dir_name = sys.argv[1]
	measure_UIQMs(dir_name)