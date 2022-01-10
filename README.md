## Deep WaveNet
#### **Wavelength-based Attributed Deep Neural Network for Underwater Image Restoration**
##### accepted in ACM Transactions on Multimedia Computing, Communications, and Applications
###### [**Prasen Kumar Sharma**](https://www.iitg.ac.in/stud/kumar176101005/), [**Ira Bisht**](), and [**Arijit Sur**](https://www.iitg.ac.in/arijit/).

###### [**Web-app**](https://deep-wavenet.herokuapp.com/) has been released (basic version). Best viewed in Firefox latest version. Note that [Heroku](https://www.heroku.com/) allows CPU-based computations only with limited memory. Hence, the app processes input image with a lower-resolution of 256x256. Use the above codes only to reproduce the original results.   

**Google Colab demo:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/DeepWaveNet_demo.ipynb)

![Block](imgs/proposed.png)

- This paper deals with the **underwater image restoration**. 
- For this, we have considered two of the main low-level vision tasks, 
  - **image enhancement**, and 
  - **super-resolution**. 
- For underwater image enhancement (uie), we have utilized publicly available datasets 
  - [**EUVP**](http://irvlab.cs.umn.edu/resources/euvp-dataset), and 
  - [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html). 
- For super-resolution, we have used [**UFO-120**](http://irvlab.cs.umn.edu/resources/ufo-120-dataset) dataset. 
- Below, we provide the detailed instructions for each task in a single README file to reproduce the original results.

[**arXiv version**](https://arxiv.org/abs/2106.07910)

### Contents
1. [**Results**](#results)
2. [**Prerequisites**](#prerequisites)
3. [**Datasets Preparation**](#datasets-preparation)
4. [**Usage**](#usage)
5. [**Evaluation Metrics**](#evaluation-metrics)
6. [**Processing underwater degraded videos**](#processing-underwater-degraded-videos)
7. [**For Underwater Semantic Segmentation and 2D pose Estimation Results**](#for-underwater-semantic-segmentation-and-2D-pose-estimation-results)
8. [**License and Citations**](#license-and-citation)
9. [**Send us feedback**](#send-us-feedback)
10. [**Acknowledgement**](#acknowledgement)
11. [**Future Releases**](#future-releases)

### Results
![Block](imgs/teaser.png)

### Prerequisites
| **Build Type**   |`Linux`           |`MacOS`           |`Windows`         |
| :---:            | :---:            | :---:            | :---:            |
| **Script**       | [env](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/installation/requirements.txt)    | [TBA]() | [TBA] |

Also, the codes work with *minimum* requirements as given below.
```bash
# tested with the following dependencies on Ubuntu 16.04 LTS system:
Python 3.5.2
Pytorch '1.0.1.post2'
torchvision 0.2.2
opencv 4.0.0
scipy 1.2.1
numpy 1.16.2
tqdm
```
To install using linux [env](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/installation/requirements.txt)
```bash
pip install -r requirements.txt

```

### Datasets Preparation
##### To test the Deep WaveNet on [EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset) dataset
```bash
cd uie_euvp
```
- Download the [**test_samples**](https://drive.google.com/drive/folders/14mRov8a2z1K2TGApRaiWL0-rGv04RUwi) from [**EUVP**](http://irvlab.cs.umn.edu/resources/euvp-dataset) dataset in the current directory (preferably). 
-If you have downloaded somewhere else on the machine, set the absolute path of the test-set arguments in [**options.py**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/uie_euvp/options.py)
```bash
---testing_dir_inp
---testing_dir_gt
```

##### To test the Deep WaveNet on [**UIEB and Challenging-60**](https://li-chongyi.github.io/proj_benchmark.html) datasets
```bash
cd uie_uieb
```
- Download the [**UIEB and Challenging-60**](https://li-chongyi.github.io/proj_benchmark.html) datasets in the current directory (preferably). 
- If you have downloaded somewhere else on the machine, set the absolute path of the test-set arguments in [**options.py**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/uie_uieb/options.py)
```bash
---testing_dir_inp
---testing_dir_gt
``` 
- The UIEB test set used in our paper is available at [**Google drive**](https://drive.google.com/drive/folders/1wlitOSdb9Q5vB596YAy9w6u10iqCudgl?usp=sharing). 
- Set the above arguments for this too if utilized.
- Note, the `challenging-60` set may not have ground truth clean images. So for that, you can leave the argument `--testing_dir_gt` blank.
- Ours results on `challenging-60` set are available at [**Google drive**](https://drive.google.com/drive/folders/1rZImxd4JHc-Idju0--LjBj135Yu_djDl?usp=sharing).

##### To test the Deep WaveNet on [**UFO-120**](http://irvlab.cs.umn.edu/resources/ufo-120-dataset) dataset
```bash
cd super-resolution
```
- For super-resolution, we have provided separate code files for following configs: `2X`, `3X`, and `4X`.
- To test on `2X`, use the following steps:
```bash
cd 2X
```
- Download the [**UFO-120**](https://drive.google.com/drive/folders/1TI5iona5Cuv6G4cfo0O4-_pPbu_QdaF1) test-set in the current directory (preferably). 
- If you have already downloaded somewhere else on the machine, set the absolute path of the test-set arguments in [**options.py**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/super-resolution/2X/options.py)
```bash
---testing_dir_inp
---testing_dir_gt
``` 
- `lrd` consists of lower-resolution images, whereas `hr` consists of corresponding high-resolution images.
- Repeat the above steps for remaining SR configs: `3X`, and `4X`.

##### For training
- Similar to test-sets, define training-set folders absolute path in the respective `options.py` file for each of the datasets above.

### Usage
##### For testing
- Once the test-sets as described [**above**](#datasets-preparation) are set.
- You can test a model for a given task using the following command:
```bash
export CUDA_VISIBLE_DEVICES=0 #[optional]
python test.py  
```
- Results will be saved in `facades` folder of the `pwd`.

##### For training
- Once the training-sets as described [**above**](#datasets-preparation) are set.
- You can train a model for a given task using the following command:
```bash
export CUDA_VISIBLE_DEVICES=0 #[optional]
python train.py --checkpoints_dir --batch_size --learning_rate             
```
- `Models` will be saved in `--checkpoints_dir` with naming convention `netG_[epoch].pt`.
- Demo codes for plotting loss curve during training are provided in [**utils/loss**]() folder.

### Evaluation Metrics
- Image quality metrics (IQMs) used in this work are provided with both [**Python**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/tree/main/utils/python_iqms) and [**Matlab**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/tree/main/utils/matlab_iqms).
- Thanks to [**Funie-GAN**](https://github.com/xahidbuffon/FUnIE-GAN/tree/master/Evaluation) for providing the python implementations of IQMs.
- Sample usage of IQMs in python implementation is provided at [**Line**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/d9edc4f5ca0489b720e0f189dc102b1df7dfb775/super-resolution/3X/test.py#L75). 
```python
  ### compute SSIM and PSNR
  SSIM_measures, PSNR_measures = SSIMs_PSNRs(CLEAN_DIR, result_dir)
  print("SSIM on {0} samples".format(len(SSIM_measures))+"\n")
  print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures))+"\n")
  print("PSNR on {0} samples".format(len(PSNR_measures))+"\n")
  print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures))+"\n")
  measure_UIQMs(result_dir)
```
- Use the same for **EUVP** and **UFO-120**.
- For **UIEB**, we have utlized matlab implementations. Before running [**ssim_psnr.m**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/utils/matlab_iqms/ssim_psnr.m), set clean path at [**Line1**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/b0d25a735a82ced1cf5678f041a209888e3ca25a/utils/matlab_iqms/ssim_psnr.m#L1), [**Line2**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/b0d25a735a82ced1cf5678f041a209888e3ca25a/utils/matlab_iqms/ssim_psnr.m#L14) and result path at [**Line**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/b0d25a735a82ced1cf5678f041a209888e3ca25a/utils/matlab_iqms/ssim_psnr.m#L13).
- Below are the results you should get for EUVP dataset

| **Method**   | `MSE` | `PSNR` | `SSIM` |
| ------------ | :---: | :----: | :----: |
| Deep WaveNet |  .29  | 28.62  |  .83   |


- Below are the results you should get for UIEB dataset

| **Method**   |`MSE`           |`PSNR`           |`SSIM`         |
| :---:            | :---:            | :---:            | :---:            |
| Deep WaveNet | .60 | 21.57 | .80 |

- Below are the results you should get for UFO-120 dataset

| **Method**   |`PSNR`           |`SSIM`           |`UIQM`         |
| :---:            | :---:            | :---:            | :---:            |
| Deep WaveNet (`2X`) | 25.71 | .77 | 2.99 |
| Deep WaveNet (`3X`) | 25.23 | .76 | 2.96 |
| Deep WaveNet (`4X`) | 25.08 | .74 | 2.97 |

### Processing underwater degraded videos
- We have also provided the sample [**codes**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/tree/main/uw_video_processing) for processing the degraded underwater videos. 
- To run, use the following steps:
```bash
cd uw_video_processing
```
- Download the degraded underwater video in `pwd`. 
- To demonstrate an *e.g.*, we have used a copyright-free [**sample**](https://www.youtube.com/watch?v=iSTHMiGeX6U) degraded video, which is provided as `degraded_video.mp4` in `pwd`.
- To test custom degraded video, replace the video path at [**line**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/59cbfcba521263e7cf01f3a82924ffa5198540cb/uw_video_processing/test.py#L80).
- For the output enhanced video path, set [**line**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/59cbfcba521263e7cf01f3a82924ffa5198540cb/uw_video_processing/test.py#L115).
- To run, execute 
```bash
python test.py
```

### For Underwater Semantic Segmentation and 2D pose Estimation Results
- To generate segmentation maps on enhanced images, follow [**SUIM**](https://github.com/xahidbuffon/SUIM). 
  - Post-processing codes related to this are available in [**utils**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/tree/main/utils) folder.
- To generate 2D pose of the human divers in an enhanced underwater image, follow [**OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

### License and Citation
- The usage of this software is only for academic purposes. One can not use it for commercial products in any form. 
- If you use this work or codes (for academic purposes only), please cite the following:
```
@misc{sharma2021wavelengthbased,
      title={Wavelength-based Attributed Deep Neural Network for Underwater Image Restoration}, 
      author={Prasen Kumar Sharma and Ira Bisht and Arijit Sur},
      year={2021},
      eprint={2106.07910},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

 @article{islam2019fast,
     title={Fast Underwater Image Enhancement for Improved Visual Perception},
     author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
     journal={IEEE Robotics and Automation Letters (RA-L)},
     volume={5},
     number={2},
     pages={3227--3234},
     year={2020},
     publisher={IEEE}
}

@ARTICLE{8917818,  
    author={Li, Chongyi 
            and Guo, Chunle 
            and Ren, Wenqi 
            and Cong, Runmin 
            and Hou, Junhui 
            and Kwong, Sam 
            and Tao, Dacheng},  
    journal={IEEE Transactions on Image Processing},   
    title={An Underwater Image Enhancement Benchmark Dataset and Beyond},   
    year={2020},  
    volume={29},  
    number={},  
    pages={4376-4389},  
    doi={10.1109/TIP.2019.2955241}
}

@inproceedings{eriba2019kornia,
  author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
  title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
  booktitle = {Winter Conference on Applications of Computer Vision},
  year      = {2020},
  url       = {https://arxiv.org/pdf/1910.02190.pdf}
}

@inproceedings{islam2020suim,
  title={{Semantic Segmentation of Underwater Imagery: Dataset and Benchmark}},
  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, 
              Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE/RSJ}
}

@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```

### Send us feedback
- If you have any queries or feedback, please contact us @(**kumar176101005@iitg.ac.in** or **ibisht@iitg.ac.in**).

### Acknowledgements
- For computing resources, we acknowledge the Department of Biotechnology, Govt. of India for the financial support for the project BT/COE/34/SP28408/2018.
- Some portion of the code are adapted from [**FUnIE-GAN**](https://github.com/xahidbuffon/FUnIE-GAN). The authors greatfully acknowledge it!
- We acknowledge the support of publicly available datasets [**EUVP**](http://irvlab.cs.umn.edu/resources/euvp-dataset), [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html), and [**UFO-120**](http://irvlab.cs.umn.edu/resources/ufo-120-dataset).  

### Future Releases
- We are in the process of releasing web-app soon. [Done]
- More flexible training modules using [**visdom**](https://ai.facebook.com/tools/visdom/) to be added soon.
