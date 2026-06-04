"""Generate semi-dense RGB-X matching maps for the densification pipeline."""

import argparse
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated.*",
)

import cv2
import grounding_dino.groundingdino.datasets.transforms as gd_transforms
import numpy as np
import torch
from grounding_dino.groundingdino.util.inference import load_model as load_grounding_dino
from grounding_dino.groundingdino.util.inference import predict
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from tqdm import tqdm

from load_model import add_method_arguments, load_model_module

DEFAULT_TEXT_PROMPT = "grass. floor. road. sky. tree. wall."
DEFAULT_SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_GROUNDING_DINO_CONFIG = "./grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GROUNDING_DINO_CHECKPOINT = "./checkpoints/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


DINO_TRANSFORM = gd_transforms.Compose(
    [
        gd_transforms.RandomResize([800], max_size=1333),
        gd_transforms.ToTensor(),
        gd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_images(folder):
    """Return image paths from a folder in deterministic order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected an image folder, got: {folder}")
    return sorted(str(path) for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dino_transform(image):
    image_transformed, _ = DINO_TRANSFORM(image, None)
    return image_transformed


def build_semantic_segmenters(args):
    sam_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=DEVICE)
    sam_predictor = SAM2ImagePredictor(sam_model)
    grounding_model = load_grounding_dino(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=DEVICE,
    )
    return sam_predictor, grounding_model


def predict_semantic_mask(image_rgb, sam_predictor, grounding_model, args):
    """Predict the semantic region to sample more densely in the reference image."""
    height, width = image_rgb.shape[:2]

    sam_predictor.set_image(image_rgb)
    image_tensor = dino_transform(Image.fromarray(image_rgb))
    boxes, _, _ = predict(
        model=grounding_model,
        image=image_tensor,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=DEVICE,
    )

    if len(boxes) == 0:
        return np.zeros((height, width), dtype=bool)

    boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    autocast_context = torch.autocast(device_type=DEVICE, dtype=torch.float16) if DEVICE == "cuda" else nullcontext()
    with autocast_context:
        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks.sum(axis=0).astype(bool)


def estimate_homography(mkpts0, mkpts1):
    if len(mkpts0) < 4:
        return np.eye(3), None
    homography, inliers = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC)
    if homography is None:
        return np.eye(3), inliers
    return homography, inliers


def filter_matches(match_result, match_thr, width, height, pad):
    mkpts0 = to_numpy(match_result["mkpts0"])
    mkpts1 = to_numpy(match_result["mkpts1"])
    mconf = to_numpy(match_result["mconf"])

    if len(mkpts0) == 0:
        return mkpts0, mkpts1

    x0 = mkpts0[:, 0]
    y0 = mkpts0[:, 1]
    valid = (x0 >= pad) & (x0 < width + pad) & (y0 >= pad) & (y0 < height + pad) & (mconf > match_thr)
    return mkpts0[valid] - pad, mkpts1[valid]


def accumulate_neighborhood(canvas, canvas_count, mkpts0, mkpts1, source_gray):
    if len(mkpts0) == 0:
        return

    height, width = canvas.shape
    source_height, source_width = source_gray.shape

    idx0 = mkpts0.astype(np.int32)
    idx1 = mkpts1.astype(np.int32)
    y0 = np.clip(idx0[:, 1], 0, height - 1)
    x0 = np.clip(idx0[:, 0], 0, width - 1)
    y1 = np.clip(idx1[:, 1], 0, source_height - 1)
    x1 = np.clip(idx1[:, 0], 0, source_width - 1)

    offsets = (
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    for dy, dx in offsets:
        yy0 = np.clip(y0 + dy, 0, height - 1)
        xx0 = np.clip(x0 + dx, 0, width - 1)
        yy1 = np.clip(y1 + dy, 0, source_height - 1)
        xx1 = np.clip(x1 + dx, 0, source_width - 1)
        canvas[yy0, xx0] += source_gray[yy1, xx1]
        canvas_count[yy0, xx0] += 1


def accumulate_semantic_region(canvas, canvas_count, warped_gray, semantic_mask, rng, sample_rate):
    """Randomly sample warped X-modality values inside the detected semantic mask."""
    valid_region = warped_gray != 0
    kernel = np.ones((3, 3), np.uint8)
    valid_region = cv2.erode(valid_region.astype(np.uint8), kernel, iterations=2).astype(bool)

    pos_y, pos_x = np.where(semantic_mask & valid_region)
    n_pixels = pos_y.size
    if n_pixels == 0:
        return

    n_sample = int(np.ceil(sample_rate * n_pixels))
    if n_sample == 0:
        return

    sample_idx = rng.choice(n_pixels, size=n_sample, replace=False)
    ys = pos_y[sample_idx]
    xs = pos_x[sample_idx]

    canvas[ys, xs] += warped_gray[ys, xs]
    canvas_count[ys, xs] += 1


def context_range(index, length, n_frames):
    """Select a local temporal window around the current frame index."""
    window = min(n_frames, length)
    half_window = window // 2

    start = max(0, index - half_window)
    end = start + window
    if end > length:
        end = length
        start = max(0, end - window)
    return start, end


def process_pair(
    matcher,
    image0_path,
    image1_paths,
    save_dir,
    save_name,
    central_frame,
    pair_index,
    sam_predictor,
    grounding_model,
    args,
):
    image0_rgb = load_rgb(image0_path)
    height, width = image0_rgb.shape[:2]
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas_count = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(args.seed + pair_index)

    semantic_mask = predict_semantic_mask(image0_rgb, sam_predictor, grounding_model, args)
    cv2.imwrite(str(save_dir / save_name), (semantic_mask.astype(np.uint8) * 255))

    for frame_idx, image1_path in enumerate(image1_paths):
        image1_rgb = load_rgb(image1_path)
        image1_gray = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2GRAY)

        match_result = matcher.from_paths(image0_path, image1_path)
        mkpts0, mkpts1 = filter_matches(match_result, args.match_thr, width, height, args.pad)
        homography, _ = estimate_homography(mkpts0, mkpts1)

        if frame_idx == central_frame:
            warped_gray = cv2.warpPerspective(
                image1_gray,
                homography,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            accumulate_semantic_region(
                canvas,
                canvas_count,
                warped_gray,
                semantic_mask,
                rng,
                args.region_sample_rate,
            )

        accumulate_neighborhood(canvas, canvas_count, mkpts0, mkpts1, image1_gray)

        eq_match_result = matcher.from_paths_eq0(image0_path, image1_path)
        eq_mkpts0, eq_mkpts1 = filter_matches(
            eq_match_result,
            args.match_thr,
            width,
            height,
            args.pad,
        )
        accumulate_neighborhood(canvas, canvas_count, eq_mkpts0, eq_mkpts1, image1_gray)

    active_area = canvas_count >= 1
    canvas[active_area] /= canvas_count[active_area]

    stem = Path(image0_path).stem
    if args.verbose:
        tqdm.write(f"{stem}: {int(active_area.sum())} final points")

    np.save(save_dir / f"{stem}.npy", canvas)
    cv2.imwrite(str(save_dir / f"{stem}.png"), canvas.astype(np.uint8))

    masked_rgb = image0_rgb.copy()
    masked_rgb[canvas_count == 0] = 0
    masked_bgr = masked_rgb[:, :, [2, 1, 0]]
    cv2.imwrite(str(save_dir / f"{stem}_masked.png"), masked_bgr.astype(np.uint8))


def run_semidense_matching(args):
    image0_list = collect_images(args.fold1)
    image1_list = collect_images(args.fold2)
    if len(image0_list) == 0:
        raise ValueError(f"No images found in --fold1: {args.fold1}")
    if len(image1_list) == 0:
        raise ValueError(f"No images found in --fold2: {args.fold2}")
    if len(image0_list) != len(image1_list):
        raise ValueError(
            f"--fold1 and --fold2 must contain the same number of images; "
            f"found {len(image0_list)} and {len(image1_list)}."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    matcher = load_model_module(args.method, args)
    sam_predictor, grounding_model = build_semantic_segmenters(args)

    for index, image0_path in enumerate(tqdm(image0_list, desc="semidense matching")):
        start, end = context_range(index, len(image1_list), args.context_frames)
        image1_context = image1_list[start:end]
        central_frame = index - start
        save_name = Path(image0_path).name
        process_pair(
            matcher,
            image0_path,
            image1_context,
            save_dir,
            save_name,
            central_frame,
            index,
            sam_predictor,
            grounding_model,
            args,
        )


def build_parser():
    method_choices = ["xoftr", "sp_lg", "loftr", "roma"]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--method", type=str, choices=method_choices, default="xoftr")
    known_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Generate semi-dense RGB-X matching maps.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=method_choices,
        help="Matcher to use.",
    )
    parser.add_argument(
        "--match_thr",
        type=float,
        default=0.1,
        help="Confidence threshold for sparse matches.",
    )

    add_method_arguments(parser, known_args.method)

    parser.add_argument("--fold1", type=str, required=True, help="RGB/reference image folder.")
    parser.add_argument("--fold2", type=str, required=True, help="X-modality image folder.")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for masks, maps, and previews.",
    )
    parser.add_argument(
        "--region_sample_rate",
        type=float,
        default=0.05,
        help="Fraction of detected semantic-region pixels sampled from the warped X image.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=16,
        help="Padding used by the matching wrapper.",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=12,
        help="Number of neighboring X frames to match.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for semantic-region sampling.",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="GroundingDINO text prompt.",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="GroundingDINO box threshold.",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="SAM2 checkpoint.",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default=DEFAULT_SAM2_MODEL_CONFIG,
        help="SAM2 model config.",
    )
    parser.add_argument(
        "--grounding_dino_config",
        type=str,
        default=DEFAULT_GROUNDING_DINO_CONFIG,
        help="GroundingDINO model config.",
    )
    parser.add_argument(
        "--grounding_dino_checkpoint",
        type=str,
        default=DEFAULT_GROUNDING_DINO_CHECKPOINT,
        help="GroundingDINO checkpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image matching statistics.",
    )
    return parser


def validate_args(args):
    if not 0.0 <= args.region_sample_rate <= 1.0:
        raise ValueError("--region_sample_rate must be between 0 and 1.")
    if args.context_frames < 1:
        raise ValueError("--context_frames must be at least 1.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.verbose:
        print(args)

    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_semidense_matching(args)
    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
