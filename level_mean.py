import glob
import cv2
import numpy as np
import os
import argparse


def process(args):
    parent = args.parent

    folders = [args.f1, args.f2, args.f3]

    # Load all image lists dynamically
    lists = []
    for f in folders:
        img_list = sorted(glob.glob(f"{parent}/{f}/*.png"))
        if len(img_list) == 0:
            raise ValueError(f"No PNG images found in {parent}/{f}")
        lists.append(img_list)

    # Create save folder
    save_folder = f"{parent}/{args.save_folder}"
    os.makedirs(save_folder, exist_ok=True)

    for frames in zip(*lists):
        fname = frames[0].rsplit("/", 1)[-1]

        images = [cv2.imread(p, -1).astype(np.float32) for p in frames]
        I_final = np.mean(images, axis=0)

        cv2.imwrite(f"{save_folder}/{fname}", I_final.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--parent", type=str, default="")
    parser.add_argument("--f1", type=str, required=True)
    parser.add_argument("--f2", type=str, required=True)
    parser.add_argument("--f3", type=str, required=True)
    parser.add_argument("--save_folder", type=str, default="merged")

    args = parser.parse_args()
    process(args)
