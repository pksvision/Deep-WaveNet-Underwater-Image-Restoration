 files = dir("./clean_test/");
 disp(length(files));
 
 ssim_vals = [];
 psnr_vals = [];
 mse_vals = [];
 
 cnt=1;
 for j =3:length(files)

	    image_name = files(j);

	        res_img = imread(strcat("./facades/Ours_UIEB/",  image_name.name));
	        ref_img = imread(strcat("./clean_test/", image_name.name));

	        [ssimvalp, ssimmap] = ssim(res_img,ref_img);
	        [peaksnrp, snr] = psnr(res_img, ref_img);
	        mse = immse(res_img, ref_img);
	        
	        ssim_vals = [ssim_vals; ssimvalp];
	        psnr_vals = [psnr_vals; peaksnrp];
	        mse_vals =  [mse_vals; mse];
	        
	        disp(mean(ssim_vals));
	        disp(mean(psnr_vals)); 
	        disp(mean(mse_vals));

	        disp(cnt);
	        cnt= cnt+1;
 
 end