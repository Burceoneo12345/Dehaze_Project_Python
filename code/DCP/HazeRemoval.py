import cv2
import numpy as np


def dark_channel(img, size=15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


def get_atmo(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t


def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # 1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b
    return q


def dehaze(path, output=None):
    im = cv2.imread(path)
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = get_atmo(img)
    trans = get_trans(img, atom)
    # cv2.imwrite("trans_canon.jpg", trans * 255)
    trans_guided = guided_filter(trans, img_gray, 15, 0.0001)
    # cv2.imwrite("trans_guided_canon.jpg", trans_guided * 255)
    trans_guided = cv2.max(trans_guided, 0.25)
    trans = cv2.max(trans, 0.25)
    result = np.empty_like(img)
    # result_1 = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
        #  result_1[:, :, i] = (img[:, :, i] - atom) / trans + atom
    # cv2.imwrite("pic_dard1.jpg", dark_channel(img, 15) * 255)
    # cv2.imwrite("result_no_guided.jpg", result_1 * 255)
    cv2.imshow("source", img)
    cv2.imshow("result", result)
    cv2.waitKey()
    if output is not None:
        cv2.imwrite(output, result * 255)


if __name__ == '__main__':
    path_1 = 'images/fog_1.jpg'
    path_2 = 'canon.jpg'
    path_3 = 'fog_2.jpeg'
    path_4 = 'girls.jpg'
    path_6 = 'input.png'
    path_5 = 'images/my_pic.jpg'
    dehaze(path_6, 'output.jpg')
