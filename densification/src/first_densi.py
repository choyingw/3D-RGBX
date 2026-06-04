from config import args as args_config
import time
import random
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm
import glob

from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.autograd.set_detect_anomaly(True)

import utility
from model.ognidc import OGNIDC

from summary.gcsummary import OGNIDCSummary
from metric.dcmetric import DCMetric
from data import get as get_data
from loss.sequentialloss import SequentialLoss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.cuda.amp as amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain, weights_only=False)

            # new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint["epoch"] + 1

    return new_args


def test(args):
    # Network
    if args.model == "OGNIDC":
        net = OGNIDC(args)
    else:
        raise TypeError(
            args.model,
            [
                "OGNIDC",
            ],
        )
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain, weights_only=False)
        key_m, key_u = net.load_state_dict(checkpoint["net"], strict=False)

        if key_u:
            print("Unexpected keys :")
            print(key_u)

        if key_m:
            print("Missing keys :")
            print(key_m)
            raise KeyError
        print("Checkpoint loaded from {}!".format(args.pretrain))

    net = nn.DataParallel(net)
    net.eval()

    t_rgb = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    t_dep = T.Compose([T.ToTensor()])

    ori_li = sorted(glob.glob(f"{args.ori}/*.npy"))
    rgb_folder = f"{args.rgb}"
    save_path = f"{args.save_path}"

    os.makedirs(save_path, exist_ok=True)
    for ori in ori_li:
        fname = ori.split("/")[-1][:-4]
        try:
            rgb_ = Image.open(f"{rgb_folder}/{fname}.png")  # .resize((518, 294), Image.Resampling.LANCZOS)
        except:
            rgb_ = Image.open(f"{rgb_folder}/{fname}.jpg")
        rgb = t_rgb(rgb_)
        dep_sp = np.load(ori)

        # background_mask = (dep_sp == 0).astype(np.uint8)
        # # Dilate the background region a bit
        # kernel = np.ones((7, 7), np.uint8)  # dilation kernel, change size as needed
        # dilated_bg = cv2.dilate(background_mask, kernel, iterations=1).astype(np.uint8)
        # dep_sp[dilated_bg == 1] = 0

        # mask = dep_sp != 0
        # idx = np.where(mask)
        # nonzero_flat_indices = np.ravel_multi_index(idx, dep_sp.shape)
        # selected = np.random.choice(nonzero_flat_indices, size=int(len(nonzero_flat_indices) * 0.99), replace=False)
        # dep_sp.flat[selected] = 0

        # dep_sp = cv2.resize(dep_sp.astype(np.uint8), (518, 294), interpolation=cv2.INTER_NEAREST)
        dep_sp = dep_sp.astype(np.float32)
        dep_sp = t_dep(dep_sp)

        rgb = rgb[None].cuda()
        dep_sp = dep_sp[None].cuda()

        K = torch.eye(3).reshape(1, 3, 3).cuda()

        sample = {
            "rgb": rgb,
            "dep": dep_sp,
            "K": K,  # dummy one; not actually used
            "pattern": 0,  # dummy one; not actually used
        }

        rgb = sample["rgb"]
        dep = sample["dep"]

        rgb_raw = torch.clone(rgb)
        dep_raw = torch.clone(dep)

        _, _, H, W = rgb.shape
        diviser = int(4 * 2 ** (args.num_resolution - 1))
        if not H % diviser == 0:
            H_new = (H // diviser + 1) * diviser
            H_pad = H_new - H
            rgb = torch.nn.functional.pad(rgb, (0, 0, 0, H_pad))
            dep = torch.nn.functional.pad(dep, (0, 0, 0, H_pad))
        else:
            H_new = H
            H_pad = 0

        if not W % diviser == 0:
            W_new = (W // diviser + 1) * diviser
            W_pad = W_new - W
            rgb = torch.nn.functional.pad(rgb, (0, W_pad, 0, 0))
            dep = torch.nn.functional.pad(dep, (0, W_pad, 0, 0))
        else:
            W_new = W
            W_pad = 0

        ### NORMAL
        # dep = dep.repeat(1, 3, 1, 1)

        sample["rgb"] = rgb
        sample["dep"] = dep

        with torch.no_grad():
            output = net(sample)

        output["pred"] = output["pred"][..., : H_new - H_pad, : W_new - W_pad]
        output["pred_inter"] = [pred[..., : H_new - H_pad, : W_new - W_pad] for pred in output["pred_inter"]]
        sample["rgb"] = rgb_raw
        sample["dep"] = dep_raw

        depth_pred = output["pred"].squeeze().cpu().numpy()

        # !!! HERE
        # depth_pred = depth_pred.transpose(1, 2, 0)
        # depth_pred = depth_pred[:, :, [2, 1, 0]]

        depth_pred = np.clip(depth_pred, 0, 255)
        cv2.imwrite(f"{save_path}/{fname}.png", depth_pred.astype(np.uint8))

        # !!! HERE
        # cmap = "jet"
        # cm = plt.get_cmap(cmap)
        # norm = plt.Normalize(vmin=np.percentile(depth_pred, 5), vmax=np.percentile(depth_pred, 95))

        # depth_colormap = cm(norm(depth_pred))
        # depth_colormap_uint8 = (depth_colormap * 255).astype(np.uint8)
        # depth_pred_colorized = Image.fromarray(depth_colormap_uint8[..., :3])

        # np.save(f"../results/{fname}.png", depth_pred)
        # depth_pred_colorized.save(f"../results/{fname}.png")


def main(args):
    init_seed()
    test(args)


if __name__ == "__main__":
    args_main = check_args(args_config)

    print("\n\n=== Arguments ===")
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ":", getattr(args_main, key), end="  |  ")
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print("")
    print("\n")

    main(args_main)
