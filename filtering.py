"""Filter sparse/dense RGB-X maps with a matcher confidence mask."""

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from load_model import load_model_module


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
METHOD_CHOICES = ("xoftr", "sp_lg")


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_images(folder):
    """Return image paths from a folder in deterministic order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected an image folder, got: {folder}")
    return sorted(
        str(path) for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_mask(mask, output_shape):
    """Resize a matcher confidence mask to image height/width."""
    height, width = output_shape[:2]
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D confidence mask, got shape {mask.shape}.")
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def filter_pair(matcher, rgb_path, target_path, save_dir, verbose=False):
    """Apply the matcher confidence mask for one RGB/target pair."""
    target = load_image(target_path)
    match_result = matcher.from_paths_filtering(rgb_path, target_path)
    conf_mask = resize_mask(to_numpy(match_result["conf_mask"]), target.shape)

    filtered = target.copy()
    filtered[conf_mask == 0] = 0

    stem = Path(rgb_path).stem
    np.save(save_dir / f"{stem}.npy", filtered.astype(np.float32))
    cv2.imwrite(str(save_dir / f"{stem}.png"), filtered.astype(np.uint8))

    if verbose:
        valid_pixels = int(np.count_nonzero(conf_mask))
        tqdm.write(f"{stem}: kept {valid_pixels} pixels")


def run_filtering(args):
    rgb_images = collect_images(args.fold1)
    target_images = collect_images(args.fold2)

    if len(rgb_images) == 0:
        raise ValueError(f"No images found in --fold1: {args.fold1}")
    if len(target_images) == 0:
        raise ValueError(f"No images found in --fold2: {args.fold2}")
    if len(rgb_images) != len(target_images):
        raise ValueError(
            f"--fold1 and --fold2 must contain the same number of images; "
            f"found {len(rgb_images)} and {len(target_images)}."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    matcher = load_model_module(args.method, args)
    if not hasattr(matcher, "from_paths_filtering"):
        raise ValueError(f"Matcher {args.method!r} does not support filtering.")

    pairs = zip(rgb_images, target_images)
    for rgb_path, target_path in tqdm(pairs, total=len(rgb_images), desc="filtering"):
        filter_pair(matcher, rgb_path, target_path, save_dir, verbose=args.verbose)


def add_filter_method_arguments(parser, method):
    """Add matcher-specific arguments while preserving legacy filtering defaults."""
    if method == "xoftr":
        parser.add_argument("--match_threshold", type=float, default=0.5)
        parser.add_argument("--fine_threshold", type=float, default=0.1)
        parser.add_argument("--ckpt", type=str, default="./checkpoints/weights_xoftr_640.ckpt")
    elif method == "sp_lg":
        parser.add_argument("--ckpt", type=str, default="./checkpoints/minima_lightglue.pth")
    else:
        raise ValueError(f"Unknown method: {method}")


def build_parser():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--method", type=str, choices=METHOD_CHOICES, default="xoftr")
    known_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Filter RGB-X maps with matcher confidence masks."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=METHOD_CHOICES,
        help="Matcher to use for confidence-mask filtering.",
    )
    add_filter_method_arguments(parser, known_args.method)

    parser.add_argument("--fold1", type=str, required=True, help="RGB/reference image folder.")
    parser.add_argument("--fold2", type=str, required=True, help="RGB-X map folder to filter.")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output folder for filtered maps.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image filtering statistics.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.verbose:
        print(args)

    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_filtering(args)
    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
