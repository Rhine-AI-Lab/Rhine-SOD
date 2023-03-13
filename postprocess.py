import argparse
import os.path as osp
from glob import glob
from time import time

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--original_path', type=str, default=r"./inputs/12.png")
parser.add_argument('--mask_path', type=str, default=r"./outputs/mask/")
parser.add_argument('--outputs_path', type=str, default="./outputs")

parser.add_argument('--thres', type=int, default=0)
parser.add_argument('--cdr', type=int, default=0, help="-1:None, 0:Larger reservation, 1:Maximum retention, 2: Center retention")
args = parser.parse_args()

img_list = glob(f"{args.original_path}")

time_sum = 0


def filter_dot(img):
    rn = img.shape[0] * img.shape[1] / 1000

    # filter black dot (Calculation of connected domain of inverse color graph)
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(np.abs(img - 255), connectivity=4)
    for i in range(retval):
        if stats[i][-1] <= rn:
            img[labels == i] = 255

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

    # filter white dot
    if args.cdr == 0:
        for i in range(retval):
            if stats[i][-1] <= rn:
                img[labels == i] = 0

    # preserve the maximum connected domain
    elif args.cdr == 1:
        n_list = []
        for i in range(retval):
            n_list.append(stats[i][-1])
        img[labels != n_list.index(max(n_list[1:]))] = 0

    elif args.cdr == 2:
        distance_list = []
        center_x, center_y = img.shape[0] / 2 , img.shape[1] / 2
        for i in range(retval):
            centroid_x, centroid_y = centroids[i]
            distance = (centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2
            distance_list.append(distance)
        img[labels != distance_list.index(min(distance_list[1:]))] = 0
    return img


for idx in tqdm(range(len(img_list))):

    time_start = time()

    name = osp.basename(img_list[idx])[:-4]
    original = cv2.imread(img_list[idx])
    mask0 = cv2.imread(osp.join(args.mask_path, name + '.png'), 0)

    if mask0 is None:
        print("Please run inference before postprocessing!")
        exit(0)

    r, g, b = cv2.split(original)

    if args.thres != 0:
        mask1 = cv2.threshold(mask0, args.thres, 255, cv2.THRESH_BINARY)[1]
        mask2 = filter_dot(mask1.copy())
        mask3 = cv2.GaussianBlur(mask2.copy(), (25, 25), 0, 0)
        mask3 = cv2.threshold(mask3, 140, 255, cv2.THRESH_BINARY)[1]
        maskf = filter_dot(mask3.copy())

    else:
        maskf = mask0

    result = cv2.merge([r, g, b, maskf])

    original = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
    if args.thres != 0:
        mask0 = cv2.cvtColor(mask0, cv2.COLOR_GRAY2RGBA)
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGBA)
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGBA)
        mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2RGBA)

    maskf = cv2.cvtColor(maskf, cv2.COLOR_GRAY2RGBA)
    if args.thres != 0:
        compare = np.hstack([original, mask0, mask1, mask2, mask3, maskf, result])
    else:
        compare = np.hstack([original, maskf, result])

    cv2.imwrite(osp.join(args.outputs_path, "result", name + '.png'), result)
    cv2.imwrite(osp.join(args.outputs_path, "compare", name + '.png'), compare)

    time_sum = time_sum + (time() - time_start)

print(time_sum)
