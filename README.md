# Zero-Shot Image Dehazing (ZID) ![]( https://visitor-badge.glitch.me/badge?page_id=liboyun.zid)

Pytorch implementation of ZID (TIP 2020) [[paper](https://ieeexplore.ieee.org/document/9170880/)]

## Dependencies

* Python == 3.6.10
* Pytorch == 1.1.0 
* opencv-python == 3.4.2.16 
* opencv-contrib-python == 3.4.2.16 

We also export our conda virtual environment as ZID.yaml. You can use the following command to create the environment.

```bash
conda env create -f ZID.yaml
```

## Demo

You can use the following command to dehaze the test image in ./data:

```bash
python dehazing.py
```

If you want to test ZID on a real world image which does not have ground truth. You can use the following command:

```bash
python RW_dehazing.py
```

The only difference between two command is whether the program calculates PSNR and SSIM.

## Citation

If you find ZID useful in your research, please consider citing:

```
@article{ZID,
author = {Li, Boyun and Gou, Yuanbiao and Liu, Jerry Zitao and Zhu, Hongyuan and Zhou, Joey Tianyi and Peng, Xi},
title = {{Zero-Shot Image Dehazing}},
journal = {IEEE Transactions on Image Processing},
year = {2020},
volume = {29},
pages = {8457--8466},
month = aug
}
```