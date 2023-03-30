import math as ma
import os
import numpy as np
import pandas as pd


def RotationMatrix(p: np.double, w: np.double, k: np.double) -> np.array:
    """
    :param p:
    :param w:
    :param k:
    :return:
    """
    # [3] -> [3 3]
    Rp = np.array([[ma.cos(p), 0, -ma.sin(p)],
                   [0, 1, 0],
                   [ma.sin(p), 0, ma.cos(p)]])

    Rw = np.array([[1, 0, 0],
                   [0, ma.cos(w), -ma.sin(w)],
                   [0, ma.sin(w), ma.cos(w)]])

    Rk = np.array([[ma.cos(k), -ma.sin(k), 0],
                   [ma.sin(k), ma.cos(k), 0],
                   [0, 0, 1]])
    return Rp @ Rw @ Rk


def P2CV_RotationMatrix(rotation: np.array) -> np.array:
    # [3 3] -> [3 3]
    rotation = np.linalg.inv(rotation)
    rotation[1:, :] = -rotation[1:, :]
    return rotation


def Transform(X: np.double, Y: np.double, Z: np.double) -> np.array:
    # [3] ->[3 1]
    return np.array([X, Y, Z]).reshape(3, -1)


def P2CV_Transform(transform: np.array, rotation: np.array) -> np.array:
    # [3 1] ->[3 1]
    return -rotation @ transform


def GenerateIntrinsicMatrix(f: np.double, x0: np.double, y0: np.double) -> np.array:
    # [3] -> [3 3]
    return np.array([f, 0, x0,
                     0, f, y0,
                     0, 0, 1]).reshape(3, 3)


def GenerateExtrinsicMatrix(rotation: np.array, T: np.array) -> np.array:
    """
    [3 3] , [3 1] -> [4 4]
    :param rotation:
    :param T:
    :return:
    """
    return np.hstack([
        np.vstack([rotation, np.array([0, 0, 0])]),
        np.vstack([T, np.array([1])])
    ])


if __name__ == '__main__':
    cams_pd = pd.read_csv("./XZTotalCam.txt", sep=" ", index_col=0)
    cams = cams_pd.to_numpy()
    ori = cams[0, 1:]
    for item in cams:
        photoName = item[0].astype(np.int32)
        item = item[1:]
        rotation = RotationMatrix(item[3], item[4], item[5])
        rotation = P2CV_RotationMatrix(rotation)
        t = P2CV_Transform(Transform(item[0] - ori[0], item[1] - ori[1], item[2] - ori[2]), rotation)
        with open(os.path.join('./cam/', str(photoName) + '.txt'), 'w') as file:
            file.write("extrinsic\n")
            np.savetxt(file, GenerateExtrinsicMatrix(rotation, t), fmt="%.6f")
            file.write("\n")
            # 像主点偏差 0.5 【-0.5】
            file.write("11748.353500 5645.437667 8639.320000\n")
            file.write("\n")
            file.write("3400 523.718750 0.100000\n")
            file.write("009_52 1080 1080 0 0 384 384")
