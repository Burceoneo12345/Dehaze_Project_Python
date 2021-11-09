import numpy as np
import math
from skimage import io, measure

tar_file_name = './picture/GridDehazeNet_result/'
ori_file_name = './picture/haze/'


def PSNR(target, ref):
    # 将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
        return 20 * math.log10(255.0 / rmse)


if __name__ == "__main__":
    val_list = './picture/test_list.txt'
    with open(val_list) as f:
        contents = f.readlines()
        haze_names = [i.strip() for i in contents]
    pic_1 = io.imread(ori_file_name + haze_names[0])
    pic_2 = io.imread(tar_file_name + 'Dehaze_' + haze_names[0])
    ssim = measure.compare_ssim(pic_1, pic_2, data_range=255, multichannel=True)
    psnr = measure.compare_psnr(pic_1, pic_2)
    print(ssim)
    print(psnr)