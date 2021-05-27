import numpy as np
from scipy import signal
import os
import cv2
import copy
from tqdm import tqdm


def getEnergyMap(src, mask=None):
    filter_x = np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.],
    ])
    filter_y = np.array([
        [-1., -2., -1.],
        [0.,  0.,  0. ],
        [1.,  2.,  1. ],
    ])
    grad_x = abs(signal.convolve2d(np.sum(src, -1), filter_x, boundary='symm', mode='same'))
    grad_y = abs(signal.convolve2d(np.sum(src, -1), filter_y, boundary='symm', mode='same'))

    if mask is None:
        mask_val = np.zeros_like(grad_x)
    else:
        print(mask.shape)
        mask_sum = np.sum(mask, -1)
        mask_val = np.where(
            mask_sum == 0,
            np.ones_like(mask_sum) * -10000.,
            np.zeros_like(mask_sum)
        )
    return grad_x + grad_y + mask_val


def getSeam(origin_energy_map):
    energy_map = copy.deepcopy(origin_energy_map)
    h, w = energy_map.shape[0:2]
    dp = np.zeros_like(energy_map, dtype=np.int32)
    for i in range(1, h):
        for j in range(0, w):
            left = max(j - 1, 0)
            right = min(j + 2, w)
            idx = np.argmin(energy_map[i - 1, left:right])
            dp[i, j] = idx + left
            energy_map[i, j] += energy_map[i - 1, idx + left]

    seam_idx = []
    j = np.argmin(energy_map[-1])
    for i in range(h - 1, -1, -1):
        seam_idx.append(j)
        j = dp[i, j]

    seam_idx.reverse()
    return np.array(seam_idx)


def getSeams(energy_map, num=10):
    seams = []
    for i in range(num):
        seam = getSeam(energy_map)
        for j in range(len(seam)):
            energy_map[j, seam[j]] = 1e10
        seams.append(seam)
    return seams


def addSeams(src, seams):
    ls = [ set() for i in range(src.shape[0]) ]
    for seam in seams:
        for i in range(len(seam)):
            ls[i].add(seam[i])
    
    lines = []
    for i in range(src.shape[0]):
        line = []
        for j in range(src.shape[1]):
            if j in ls[i]:
                line.append(src[i, j])
            line.append(src[i, j])
        lines.append(line)
    return np.array(lines)


def img2square(src):
    flag = False
    if src.shape[0] < src.shape[1]:
        flag = True
        src = np.transpose(src, [1, 0, 2])

    num = src.shape[0] - src.shape[1]
    print(src.shape)
    while num > 0:
        if num >= 10:
            num -= 10
            num_p = 10
        else:
            num_p = num
            num = 0
        energy_map = getEnergyMap(src)
        seams = getSeams(energy_map, num=num_p)
        dst = addSeams(src, seams)
        src = dst

    if flag:
        src = np.transpose(src, [1, 0, 2])
    src = cv2.resize(src, (224, 224))

    return src


if __name__ == '__main__':
    path = 'data/training'
    output_path = 'data/training_224x224'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for folder in sorted(os.listdir(path)):
        label = int(folder.split('.')[0])
        if not os.path.exists(os.path.join(output_path, folder)):
            os.mkdir(os.path.join(output_path, folder))
        for filename in os.listdir(os.path.join(path, folder)):
            img = cv2.imread(os.path.join(path, folder, filename))
            cv2.imwrite(img2square(img), os.path.join(output_path, folder, filename))
