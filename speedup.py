import argparse
import os.path as osp

import torch
from nni.compression.pytorch import ModelSpeedup

from lib import RhNet_SwinB
from utils.misc import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r"./weights/RhineSOD.pth", help="weights path")
    parser.add_argument('--gpu', '-g', action='store_true', default=True)
    parser.add_argument('--config', '-c', type=str, default='configs/RhineSOD.yaml')
    parser.add_argument('--imgsize', type=int, default=320, help='input image size')
    parser.add_argument('--thres', type=int, default=50)
    parser.add_argument('--original_path', type=str, default=r"G:\ML-Dataset\DUTS-TR\images", help="input image path")
    parser.add_argument('--label_path', type=str, default=r"G:\ML-Dataset\DUTS-TR\masks", help="input image path")
    parser.add_argument('--mask_path', type=str, default="./outputs/mask", help="output masked path")
    args = parser.parse_args()

    opt = load_config(args.config)

    model = RhNet_SwinB(**opt.Model)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')), strict=True)

    model = model.cuda()


    with torch.no_grad():
        ModelSpeedup(model, torch.randn(4, 3, 512, 512).to(torch.device('cuda')),
                     "./400/best_result/masks.pth").speedup_model()

    torch.save(model.state_dict(), osp.join(opt.Train.Checkpoint.checkpoint_dir, 'compressed.pth'))
