import glob
import sys

import numpy as np
import os
import gdal

acc_dict = {'abs_err': 0,
            'acc_30cm': 0,
            'acc_60cm': 0,
            'acc_100cm': 0,
            'acc_120cm': 0,
            }


def abs_error(depth_pred, depth_gt):
    all = depth_pred.shape[0] * depth_pred.shape[1]
    error = np.abs(depth_pred - depth_gt)
    error[error > 10] = 0
    p = np.sum(error) / all
    return p


def acc_threshold(depth_pred, depth_gt, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    all = depth_pred.shape[0] * depth_pred.shape[1]
    errors = np.abs(depth_pred - depth_gt)
    acc = np.sum(errors < threshold) / all
    return acc


def evaluation(pred_dir, gt_dir):
    files = glob.glob(pred_dir + "/*.tif")
    MAE = 0
    acc_30cm = 0
    acc_60cm = 0
    acc_100cm = 0
    for file in files:
        fn = os.path.basename(file)
        pred_img = gdal.Open(file).ReadAsArray().astype("float32")
        # print(pred_img.shape)
        gt_file = os.path.join(gt_dir, fn)
        gt_img = gdal.Open(gt_file).ReadAsArray().astype("float32")
        p = abs_error(pred_img, gt_img)
        MAE += p / len(pred_dir)
        acc_30 = acc_threshold(pred_img, gt_img, 0.3)
        acc_30cm += acc_30 / len(pred_dir)
        acc_60 = acc_threshold(pred_img, gt_img, 0.6)
        acc_60cm += acc_60 / len(pred_dir)
        acc_100 = acc_threshold(pred_img, gt_img, 1)
        acc_100cm += acc_100 / len(pred_dir)
    print("均方误差为: %s" % MAE)
    print("acc_30cm的精度为: %s" % acc_30cm)
    print("acc_60cm的精度为: %s" % acc_60cm)
    print("acc_100cm的精度为: %s" % acc_100cm)


if __name__ == '__main__':
    a = r"C:\Users\zpl\Desktop\Smart3D\data\eval\a"
    b = r"C:\Users\zpl\Desktop\Smart3D\data\eval\b"
    evaluation(a, b)
