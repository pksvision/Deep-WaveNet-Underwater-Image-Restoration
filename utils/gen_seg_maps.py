# import shutil
import os 
from PIL import Image

classes = ['FV', 'HD', 'RI', 'RO', 'WR']

dirs = ['GT']

cls_colors = [(255,255,0), (0,0,255), (255,0,255), (255,0,0), (0,255,255)]		

for d in dirs:
	# get_image names
	total_imgs = os.listdir('./'+d+'_output/'+classes[0]+'/')

	for i in total_imgs:
		# read all class images for each image
		cls_imgs = []
		for c in range(len(classes)):
			src_img = './'+d+'_output/'+classes[c]+'/'+i
			t_img = Image.open(src_img)
			rgb_im = t_img.convert('RGB')
			cls_imgs.append(rgb_im)

		# get weidth height
		width, height = cls_imgs[0].size
		seg_image = Image.new('RGB', (width, height))

		for c_idx in range(len(cls_imgs)):
			# 2d iterate wxh
			for w in range(width):
				for h in range(height):
					r, g, b = cls_imgs[c_idx].getpixel((w, h))
					if (r, g, b) == (255, 255, 255):
						seg_image.putpixel((w, h), cls_colors[c_idx])

		
		dst = './SEG_MAPS/'+d+'/'
		if not os.path.exists(dst):
			os.makedirs(dst)
		seg_image.save(dst+str(i).split('.')[0]+'.png', 'png')