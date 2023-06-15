import argparse
import math
import cv2
import glob, os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as mcssim
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def preprocess_data(img_path):
    """_summary_

    Args:
        img_path (str): path to image

    Returns:
        tuple: tuple of original image and blurred gray image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blurred = cv2.blur(gray, (3, 3))
    
    return img, img_gray_blurred
    

def find_circles(img_gray_blurred):
    """_summary_

    Args:
        img_gray_blurred (np.array): an blurred gray image

    Returns:
        any: tuple of (a, b, r); in which a and b is row and column index (center point), r is the radius.
    """
    detected_circles = cv2.HoughCircles(img_gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, 
                                        param1 = 50, param2 = 30, 
                                        minRadius = 10, maxRadius = 40)
    detected_circles = detected_circles[0, detected_circles[:,:,0].argsort()]
    return detected_circles
    

def crop_img(img, detected_circles):
    """_summary_

    Args:
        img (np.array): a full-size image
        detected_circles (np.array): circles candidates
        margin (int, optional): Defaults to 2.
        
    Returns:
        any: cropped image.
    """
    if detected_circles is not None:
        first_circle = detected_circles[0, 0]
        first_circle = np.int16(np.around(first_circle))
        a, b, r = first_circle[0], first_circle[1], first_circle[2]
        x_max, y_max, _ = img.shape
        x_start, x_end = max(0, b - r), min(x_max, b + r)
        y_start, y_end = max(0, a - r), min(y_max, a + r)
        img = img[x_start:x_end, y_start:y_end]
    return img


def get_mask(img_size):
    mask = [[1 for _ in range(img_size)] for _ in range(img_size)]
    
    if img_size % 2 == 1:
        max_mask_idx = img_size // 2 + 1
    else:
        max_mask_idx = img_size // 2
    
    for row in range(max_mask_idx):
        for col in range(max_mask_idx):
            if row + col > max_mask_idx - 1:
                break
            if (row - max_mask_idx)**2 + (col - max_mask_idx)**2 > max_mask_idx**2:
                mask[row][col] = 0
                mask[img_size - row - 1][col] = 0
                mask[img_size - row - 1][img_size - col - 1] = 0
                mask[row][img_size - col - 1] = 0
    return mask

def main(ref_path, test_path, img_size=42):
    
    mask = np.array([get_mask(img_size)] * 3, dtype=np.int8)
    mask = np.transpose(mask, (1, 2, 0))

    ref, ref_labels = [], []
    with open(os.path.join(ref_path, 'metadata.txt'), 'r') as fin:
        for line in fin:
            splt = line.split()
            img = cv2.imread(os.path.join(ref_path, 'images' , splt[0]), cv2.IMREAD_COLOR)
            img = img * mask
            ref.append(img)
            ref_labels.append(splt[1])
            
    groundtruth = {}
    with open(os.path.join(test_path, 'test.txt'), 'r') as fin:
        for line in fin:
            splt = line.split()
            groundtruth[splt[0]] = splt[1]

    res_ssim, res_psnr = {}, {}
    for file in tqdm(glob.glob(f"{test_path}/test_images/*.jpg")):
        img, img_gray_blurred = preprocess_data(file)
        circles = find_circles(img_gray_blurred)
        cropped_img = crop_img(img, circles)
        cropped_img = cv2.resize(cropped_img, (img_size, img_size))
        masked_img = mask * cropped_img
            
        scores_ssim, scores_psnr = [], []
        for ith, src in enumerate(ref):
            score_ssim = tf.image.ssim(masked_img, src, max_val=1.0)
            scores_ssim.append(score_ssim)
            score_psnr = cv2.PSNR(masked_img, src)
            scores_psnr.append(score_psnr)
            # plt.imshow(src)
            # plt.show()
        
            # plt.imshow(masked_img)
            # plt.show()
            
            # raise
        pred_ssim = np.argmax(scores_ssim)
        pred_psnr = np.argmax(scores_psnr)
        file = file.split('\\')[1]
        res_ssim[file] = ref_labels[pred_ssim]
        res_psnr[file] = ref_labels[pred_psnr]

    ssim_true_samples, ssim_false_samples = 0, 0
    for sample, pred in res_ssim.items():
        gt = groundtruth[sample]
        if pred == gt:
            ssim_true_samples += 1
        else:
            ssim_false_samples += 1
    print('SSIM: ', round((ssim_true_samples)/(ssim_true_samples + ssim_false_samples) * 100, 3), '%')
    
    psnr_true_samples, psnr_false_samples = 0, 0
    for sample, pred in res_psnr.items():
        gt = groundtruth[sample]
        if pred == gt:
            psnr_true_samples += 1
        else:
            psnr_false_samples += 1
    print('PSNR: ', round((psnr_true_samples)/(psnr_true_samples + psnr_false_samples) * 100, 3), '%')
        

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('ref_path', type=str)
    # parser.add_argument('test_path', type=str)
    # args = parser.parse_args()
    # ref_path = args.ref_path
    # test_path = args.test_path
    # main(ref_path, test_path)
    
    main(
        ref_path='C:/Users/Admin/Documents/Computer Science/AI Engineer Test - HuynhNguyenHieuNghia/reference/', 
        test_path='C:/Users/Admin/Documents/Computer Science/AI Engineer Test - HuynhNguyenHieuNghia/test_data/'
    )

            