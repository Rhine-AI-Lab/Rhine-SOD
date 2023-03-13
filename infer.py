import argparse
import sys
import time
# import cv2
import torch
import tqdm

import os.path as osp
import numpy as np

from PIL import Image
from glob import glob
from torch.nn import functional as F
from torchvision import transforms

from lib.transforms import dynamic_resize, tonumpy, normalize, totensor  # 误报未引用依赖

filepath = osp.split(osp.abspath(__file__))[0]
repopath = osp.split(filepath)[0]
sys.path.append(repopath)

from utils.misc import load_config
from lib.RhNet import RhNet_SwinB

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def inference(opt, args):
    model = RhNet_SwinB(**opt.Model)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')), strict=True)

    if args.gpu is True:
        model = model.cuda()
    model.eval()

    img_list = glob(f"{args.original_path}\\*")

    tfs = opt.Infer.transforms
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    transform = transforms.Compose(comp)

    time_sum = 0

    for idx in tqdm.tqdm(range(len(img_list))):
        with open(img_list[idx], 'rb') as f:

            original = Image.open(f).convert('RGB')
            shape = original.size[::-1]
            name = osp.basename(img_list[idx])[:-4]

            inputs = {'image': original, 'name': name, 'shape': shape, 'original': original}
            inputs = transform(inputs)
            inputs['image'] = inputs['image'].unsqueeze(0)
            if 'image_resized' in inputs.keys():
                inputs['image_resized'] = inputs['image_resized'].unsqueeze(0)

            del original, shape, name

            if args.gpu:
                for key in inputs.keys():
                    if type(inputs[key]) == torch.Tensor:
                        inputs[key] = inputs[key].cuda()

            with torch.no_grad():
                time_start = time.time()
                out = model(inputs)
                time_sum = time_sum + (time.time() - time_start)

            pred = F.interpolate(out['pred'], inputs['shape'], mode='bilinear', align_corners=True)
            pred = pred.data.cpu().numpy().squeeze()

            # r, g, b = cv2.split(np.array(inputs['original']))
            pred = (pred * 255).astype(np.uint8)
            # if args.thres != 0:
            #     pred = cv2.threshold(pred, args.thres, 255, cv2.THRESH_BINARY)[1]
            # result = cv2.merge([r, g, b, pred]).astype(np.uint8)

            Image.fromarray(pred).save(osp.join(args.mask_path, inputs['name'] + '.png'))
            # Image.fromarray(result).save(osp.join(args.mask_path, 'result', inputs['name'] + '.png'))

    print(f"time_sum: {time_sum}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r"./weights/RhineSOD.pth", help="weights path")
    parser.add_argument('--gpu', '-g', action='store_true', default=True)
    parser.add_argument('--config', '-c', type=str, default='configs/RhineSOD.yaml')
    parser.add_argument('--imgsize', type=int, default=320, help='input image size')
    parser.add_argument('--thres', type=int, default=50)
    parser.add_argument('--original_path', type=str, default=r"./inputs", help="input image path")
    parser.add_argument('--mask_path', type=str, default="./outputs/mask", help="output masked path")
    args = parser.parse_args()

    opt = load_config(args.config)
    inference(opt, args)
