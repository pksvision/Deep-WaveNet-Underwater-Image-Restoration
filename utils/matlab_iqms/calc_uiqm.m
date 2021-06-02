 files = dir("./results/");
 disp(length(files));
 
 uiqm_vals = [];
 
 cnt=1;
 for j =3:length(files)
     
        image_name = files(j);
        res_img = imread(strcat("./results/",  image_name.name));
        uiq = UIQM(res_img);
        
        uiqm_vals = [uiqm_vals; uiq];
        
        disp(mean(uiqm_vals));
        disp(cnt);
        cnt= cnt+1;
 end
 
 
 
 
 
 
 