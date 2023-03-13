import argparse
import os.path as osp
import random
import sys
import time
from glob import glob

import nni
import numpy as np
import torch
import tqdm
from PIL import Image
from nni.algorithms.compression.v2.pytorch import TorchEvaluator
from nni.algorithms.compression.v2.pytorch.pruning import AutoCompressPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from torch.nn import ParameterList
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms

from lib.RhNet import RhNet_SwinB
from utils.eval import evaluate_acc
from utils.misc import load_config
from lib.transforms import dynamic_resize, tonumpy, normalize, totensor  # 误报未引用依赖

filepath = osp.split(osp.abspath(__file__))[0]
repopath = osp.split(filepath)[0]
sys.path.append(repopath)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def criterion(input, target):
    return input['loss']


def training_func(model, optimizer, criterion, lr_schedulers=None, max_steps=None, max_epochs=None):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Starting Training.")
    # global opt, args
    #
    # train_dataset = RGB_Dataset(
    #     root="G:/ML-Dataset/",
    #     sets=["DUTS-TR"],
    #     tfs=opt.Train.Dataset.transforms)
    #
    # train_sampler = None
    #
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=opt.Train.Dataloader.batch_size,
    #                                            shuffle=True,
    #                                            sampler=train_sampler,
    #                                            num_workers=opt.Train.Dataloader.num_workers,
    #                                            pin_memory=opt.Train.Dataloader.pin_memory,
    #                                            drop_last=True)
    #
    # scaler = None
    #
    # scheduler = PolyLr(optimizer, gamma=0.9,
    #                    minimum_lr=1.0e-07,
    #                    max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,
    #                    warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    #
    # model.train()
    # start = 1
    #
    # for epoch in range(start, opt.Train.Scheduler.epoch + 1):
    #     step_iter = enumerate(tqdm.tqdm(train_loader), start=1)
    #
    #     for i, sample in step_iter:
    #         optimizer.zero_grad()
    #
    #         sample = to_cuda(sample)
    #         loss = criterion(model(sample), None)
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()


def evaluating_func(model):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Starting evaluation.")
    model.eval()

    img_list = random.sample(glob(f"{args.original_path}\\*"), 500)

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
    acc_list = []
    for idx in tqdm.tqdm(range(len(img_list))):
        with open(img_list[idx], 'rb') as f:

            original = Image.open(f).convert('RGB')
            shape = original.size[::-1]
            name = osp.basename(img_list[idx])[:-4]
            label = np.array(Image.open(args.label_path + '\\' + name + '.png').convert('1'))

            inputs = {'image': original, 'name': name, 'shape': shape, 'original': original}
            inputs = transform(inputs)
            inputs['image'] = inputs['image'].unsqueeze(0)
            if 'image_resized' in inputs.keys():
                inputs['image_resized'] = inputs['image_resized'].unsqueeze(0)

            for key in inputs.keys():
                if type(inputs[key]) == torch.Tensor:
                    inputs[key] = inputs[key].cuda()

            with torch.no_grad():
                time_start = time.time()
                out = model(inputs)
                time_sum = time_sum + (time.time() - time_start)

            pred = F.interpolate(out['pred'], inputs['shape'], mode='bilinear', align_corners=True)
            pred = pred.data.cpu().numpy().squeeze()

            pred = (pred * 255).astype(np.uint8)
            acc = evaluate_acc(label, pred)
            acc_list.append(acc)

    print(f"Accuracy: {sum(acc_list) / len(acc_list)} ; Time: {time_sum}")
    return acc


def main(opt, args):
    model = RhNet_SwinB(**opt.Model)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')), strict=True)

    model = model.cuda()

    backbone_params = ParameterList()
    decoder_params = ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]

    config_list = [{
        'sparsity_per_layer': 0.25,
        'op_types': ['Conv2d', 'Linear']
    }, {
        'exclude': True,
        'op_names': ['backbone.patch_embed.proj']
    }]

    traced_optimizer = nni.trace(Adam)(params_list, opt.Train.Optimizer.lr,
                                       weight_decay=opt.Train.Optimizer.weight_decay)

    dummy_input = torch.randn(4, 3, 512, 512).to(torch.device('cuda'))

    evaluator = TorchEvaluator(training_func, optimizers=traced_optimizer, criterion=criterion,
                               dummy_input=dummy_input, evaluating_func=evaluating_func)

    admm_params = {
        'evaluator': evaluator,
        'iterations': 5,
        'training_epochs': 2
    }
    sa_params = {
        'evaluator': evaluator
    }

    pruner = AutoCompressPruner(model=model, config_list=config_list, total_iteration=3, admm_params=admm_params,
                                sa_params=sa_params, log_dir='./log', keep_intermediate_result=True,
                                evaluator=evaluator, speedup=True)
    # pruner = LevelPruner(model, config_list)

    pruner.compress()
    _, model, masks, _, _ = pruner.get_best_result()

    torch.save(model.state_dict(), osp.join(opt.Train.Checkpoint.checkpoint_dir, 'compressed.pth'))


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
    main(opt, args)
