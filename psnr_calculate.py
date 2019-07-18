from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import os
import matplotlib.pyplot as plt
from os import walk
from util import rgb2ycbcr,calc_metrics,ssim
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border,]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border,]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)



parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--gt_dir', default='./Vid4/walk/', help="the ground thruth dir")
parser.add_argument('--hr_dir', default='./Results/Vid4/walk_4x/', help='the high resolution dir')

# im_hr = Image.open('/home/tangrui/rrrr/RBPN-PyTorch-master/Results/Vid4/foliage_4x/1_RBPNF7.png')
# im_gt = Image.open('/home/tangrui/rrrr/RBPN-PyTorch-master/Vid4/foliage/001.png')

# im_gt_ycbcr = rgb2ycbcr(np.array(im_gt))
# im_hr_ycbcr = rgb2ycbcr(np.array(im_hr))

# im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
# im_hr_y = im_hr_ycbcr[:,:,0].astype(float)

opt = parser.parse_args()
name = []
for (dirpath, dirnames, filenames) in walk(opt.gt_dir):
    name.extend(filenames)
    break
lenth = len(name)
count = 0
PSNR_total = 0
SSIM_total = 0
for i in name:
    if 'png' in i:
        if int(i[0:-4]) < 7 :
            continue
        if int(i[0:-4]) > lenth - 3:
            continue
        hr_name = opt.hr_dir + str(int(i[0:-4])) + '_RBPNF7.png'
        gt_name = opt.gt_dir + i

        print(hr_name, gt_name)
    ###the vavalue [0,255]
        im_hr = np.array(Image.open(hr_name))
        im_gt = np.array(Image.open(gt_name))

        # im_hr_y = rgb2ycbcr(img = im_hr, only_y=True)
        # im_gt_y = rgb2ycbcr(img = im_gt, only_y=True)

        psnr, SSIM = calc_metrics(im_hr, im_gt,crop_border = 8)
        print(psnr, SSIM)
        PSNR_total += psnr
        SSIM_total += SSIM
        count += 1

print(PSNR_total,SSIM_total)
print('total_number = ',float(count))
print('average_psnr =  ',PSNR_total/float(count))
print('average_ssim = ',SSIM_total/float(count))




