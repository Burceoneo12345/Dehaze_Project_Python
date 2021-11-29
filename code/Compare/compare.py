import numpy as np
import math
from skimage import io, measure

GridDehazeNet_file_name = './picture/GridDehazeNet_result/'
DehazeNet_file_name = './picture/DehazeNet_result/'
DCP_file_name = './picture/DCP_result/'
clear_file_name = './picture/clear/'


if __name__ == "__main__":
    val_list = './picture/test_list.txt'
    ssim_DCP = []
    psnr_DCP = []
    ssim_DehazeNet = []
    psnr_DehazeNet = []
    ssim_GridDehazeNet = []
    psnr_GridDehazeNet = []
    with open(val_list) as f:
        contents = f.readlines()
        haze_names = [i.strip() for i in contents]
    for name in haze_names:
        pic_name = name.split('_')[0] + '.jpg'
        gt_name = name.split('_')[0] + '.png'
        pic_gt = io.imread(clear_file_name + gt_name)
        pic_DCP = io.imread(DCP_file_name + pic_name)
        pic_DehazeNet = io.imread(DehazeNet_file_name + pic_name)
        pic_GridDehazeNet = io.imread(GridDehazeNet_file_name + pic_name)
        ssim = measure.compare_ssim(pic_gt, pic_DCP, data_range=255, multichannel=True)
        ssim_DCP.append(ssim)
        psnr = measure.compare_psnr(pic_gt, pic_DCP)
        psnr_DCP.append(psnr)
        ssim = measure.compare_ssim(pic_gt, pic_DehazeNet, data_range=255, multichannel=True)
        ssim_DehazeNet.append(ssim)
        psnr = measure.compare_psnr(pic_gt, pic_DehazeNet)
        psnr_DehazeNet.append(psnr)
        ssim = measure.compare_ssim(pic_gt, pic_GridDehazeNet, data_range=255, multichannel=True)
        ssim_GridDehazeNet.append(ssim)
        psnr = measure.compare_psnr(pic_gt, pic_GridDehazeNet)
        psnr_GridDehazeNet.append(psnr)
    DCP = np.mean(np.array(psnr_DCP))
    DehazeNet = np.mean(np.array(psnr_DehazeNet))
    GridDehazeNet = np.mean(np.array(psnr_GridDehazeNet))
    print(DCP)
    print(DehazeNet)
    print(GridDehazeNet)