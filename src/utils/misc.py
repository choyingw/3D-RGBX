import os
import contextlib
import joblib
from typing import Union
from loguru import _Logger, logger
from itertools import chain

import torch
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import cv2


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level, lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """A temporary fix for pytorch-lighting 1.3.x"""
    gpus = str(gpus)
    gpu_ids = []

    if "," not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(",") if i != ""]

    # setup environment variables
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        logger.warning(
            f"[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}"
        )
    else:
        logger.warning("[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.")
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def project_rgb_forward_numpy(
    src_rgb: np.ndarray,  # (H, W, 3), uint8 or float
    src_depth: np.ndarray,  # (H, W), metric depth in source camera
    K_src: np.ndarray,  # (3, 3)
    K_tgt: np.ndarray,  # (3, 3)
    R: np.ndarray,  # (3, 3)  source->target rotation
    t: np.ndarray,  # (3,)    source->target translation
    tgt_shape=None,  # (H_t, W_t) if different from source
    background_color=(0, 0, 0),  # fill color for holes
):
    """
    Forward splat from source view into target view with a z-buffer.
    Nearest-neighbor splat (no antialias); fast and robust for starters.
    """
    H, W = src_depth.shape
    if tgt_shape is None:
        Ht, Wt = H, W
    else:
        Ht, Wt = tgt_shape

    # Flatten source pixels
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    Z = src_depth  # (H, W)

    # Valid depth mask (positive and finite)
    m = np.isfinite(Z) & (Z > 0)

    uu = uu[m].astype(np.float64)  # (N,)
    vv = vv[m].astype(np.float64)  # (N,)
    Z = Z[m].astype(np.float64)  # (N,)

    # Back-project to 3D in source camera
    Kinv = np.linalg.inv(K_src)
    ones = np.ones_like(uu)
    pix_h = np.stack([uu, vv, ones], axis=0)  # (3, N)
    Xs = (Kinv @ pix_h) * Z  # (3, N)

    # Transform to target camera
    Xt = (R @ Xs) + t.reshape(3, 1)  # (3, N)

    # Keep only points in front of target camera
    Zt = Xt[2, :]
    valid_z = Zt > 1e-6
    Xt = Xt[:, valid_z]
    uu = uu[valid_z]
    vv = vv[valid_z]
    Zt = Zt[valid_z]

    # Project into target pixels
    x_h = K_tgt @ Xt  # (3, N)
    u_t = x_h[0, :] / x_h[2, :]
    v_t = x_h[1, :] / x_h[2, :]

    # Nearest-neighbor splat (round to integer pixels)
    u_i = np.round(u_t).astype(np.int32)
    v_i = np.round(v_t).astype(np.int32)

    # print("pro", u_i.min(), u_i.max(), v_i.min(), v_i.max(), Wt, Ht)

    # Keep only coordinates within the target frame
    in_bounds = (u_i >= 0) & (u_i < Wt) & (v_i >= 0) & (v_i < Ht)
    u_i = u_i[in_bounds]
    v_i = v_i[in_bounds]
    Zt = Zt[in_bounds]

    # Gather corresponding colors from source
    src_rgb_f = src_rgb.astype(np.float64)
    colors = src_rgb_f[vv[in_bounds].astype(np.int64), uu[in_bounds].astype(np.int64), :]  # (M, 3)

    # Z-buffer resolve by sorting front-to-back, then picking first per pixel
    lin_idx = v_i * Wt + u_i  # (M,)
    order = np.argsort(Zt)  # small Zt (closer) first
    lin_sorted = lin_idx[order]
    col_sorted = colors[order]
    z_sorted = Zt[order]

    # Keep the *first* occurrence per pixel (nearest depth)
    # np.unique returns sorted unique keys and their first indices
    uniq_lin, uniq_first = np.unique(lin_sorted, return_index=True)

    # Initialize outputs
    tgt_rgb = np.zeros((Ht, Wt, 3), dtype=src_rgb.dtype)
    if np.issubdtype(tgt_rgb.dtype, np.integer):
        bg = np.array(background_color, dtype=tgt_rgb.dtype)
    else:
        bg = np.array(background_color, dtype=np.float64)
    tgt_rgb[:] = bg

    zbuf = np.full((Ht, Wt), np.inf, dtype=np.float64)

    # Write winning colors and depths
    win_lin = lin_sorted[uniq_first]
    win_cols = col_sorted[uniq_first]
    win_z = z_sorted[uniq_first]

    tgt_rgb.reshape(-1, 3)[win_lin] = win_cols.astype(tgt_rgb.dtype)
    zbuf.reshape(-1)[win_lin] = win_z

    return tgt_rgb, zbuf


def make_hole_mask_from_z(zbuf: np.ndarray) -> np.ndarray:
    """
    Holes are where z==inf (never written). Returns uint8 mask: 1=hole, 0=valid.
    """
    return (np.isinf(zbuf)).astype(np.uint8)


def infill_holes(
    img: np.ndarray,  # (H,W,3) uint8/float32
    hole_mask: np.ndarray,  # (H,W) uint8/bool, 1/True = hole
    mode: str = "telea",  # "telea" | "nearest" | "box"
    radius: int = 3,  # used by telea, and iteration kernel notion for box
) -> np.ndarray:
    """
    Infill disocclusion holes produced by forward warping.
    """
    mode = mode.lower()
    hole = hole_mask > 0

    if not hole.any():
        return img.copy()

    H, W = hole.shape
    out = img.copy()

    if mode == "telea":
        # OpenCV edge-aware inpainting
        try:
            import cv2
        except Exception:
            raise ImportError("telea mode requires OpenCV (cv2). Install opencv-python.")
        # cv2.inpaint expects 8-bit single-channel mask: 255=inpaint, 0=keep
        m = hole.astype(np.uint8) * 255
        # Ensure uint8 BGR for OpenCV; support both uint8 and float
        if out.dtype != np.uint8:
            tmp = np.clip(out, 0, 255).astype(np.uint8)
        else:
            tmp = out
        bgr = tmp[..., ::-1]  # RGB->BGR
        inpainted = cv2.inpaint(bgr, m, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        out = inpainted[..., ::-1]  # BGR->RGB
        # If original was float, convert back to float range
        if img.dtype != np.uint8:
            out = out.astype(img.dtype)
        return out

    elif mode == "nearest":
        # Voronoi nearest-color fill using distance transform labels (OpenCV)
        try:
            import cv2
        except Exception:
            raise ImportError("nearest mode requires OpenCV (cv2). Install opencv-python.")

        # We want distance to the nearest VALID pixel.
        # cv2.distanceTransformWithLabels needs non-zero as "object" (valids),
        # so invert the mask: valid=1, holes=0
        valid = (~hole).astype(np.uint8)
        # Distance transform from holes to valids with labels
        dist, labels = cv2.distanceTransformWithLabels(
            valid, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL
        )
        # labels are 1..N for valid pixels, and 0 in holes where no label (shouldn’t happen if there’s any valid)
        # Recover coordinates of each valid pixel index
        ys, xs = np.mgrid[0:H, 0:W]
        # Flatten valid indices
        flat_idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
        # Map each pixel to its nearest valid pixel’s flat index:
        # Build lookup from label -> flat_index of that valid pixel.
        # A label corresponds to the position of the seed the DT grew from; OpenCV’s label equals 1+seed_index in raster scan.
        # We can reconstruct the seed map by setting each valid pixel’s label to its (1+flat_index), then running DT again.
        # More simply: create an array seeds where seeds[y,x] = 1+flat_index for valid pixels, else 0, and use labels as indices.

        seeds = np.zeros((H, W), dtype=np.int32)
        seeds[~hole] = flat_idx[~hole] + 1  # 1-based ids
        # Now labels[y,x] gives the 1-based id of nearest valid pixel; convert to flat index:
        nearest_flat = np.clip(labels - 1, 0, H * W - 1)

        # Pull colors from nearest valid pixel
        src = img.reshape(-1, img.shape[-1])
        out = img.copy()
        # Only fill hole pixels
        hole_flat = flat_idx[hole]
        out.reshape(-1, img.shape[-1])[hole_flat] = src[nearest_flat[hole]]
        return out

    elif mode == "box":
        # Pure NumPy iterative neighbor averaging (fast, no deps).
        # Repeatedly average 4-neighborhood of known pixels into holes.
        out = img.astype(np.float32, copy=True)
        known = (~hole).astype(np.uint8)
        iters = max(3, radius * 2)  # heuristic
        for _ in range(iters):
            # 4-neighborhood sums
            sum_img = np.zeros_like(out, dtype=np.float32)
            cnt = np.zeros((H, W, 1), dtype=np.float32)

            # up
            sum_img[1:] += out[:-1]
            cnt[1:] += known[:-1][..., None]
            # down
            sum_img[:-1] += out[1:]
            cnt[:-1] += known[1:][..., None]
            # left
            sum_img[:, 1:] += out[:, :-1]
            cnt[:, 1:] += known[:, :-1][..., None]
            # right
            sum_img[:, :-1] += out[:, 1:]
            cnt[:, :-1] += known[:, 1:][..., None]

            # New fills only where we still have holes and have at least one known neighbor
            can_fill = (hole) & (cnt[..., 0] > 0)
            out[can_fill] = sum_img[can_fill] / cnt[can_fill]
            known[can_fill] = 1
            hole[can_fill] = 0
            if not hole.any():
                break

        # Cast back to original dtype
        if img.dtype == np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = out.astype(img.dtype)
        return out

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'telea', 'nearest', or 'box'.")


def infill_holes_1chan(
    img: np.ndarray,  # (H,W,3) uint8/float32
    hole_mask: np.ndarray,  # (H,W) uint8/bool, 1/True = hole
    mode: str = "telea",  # "telea" | "nearest" | "box"
    radius: int = 3,  # used by telea, and iteration kernel notion for box
) -> np.ndarray:
    """
    Infill disocclusion holes produced by forward warping.
    """
    mode = mode.lower()
    hole = hole_mask > 0

    if not hole.any():
        return img.copy()

    H, W = hole.shape
    out = img.copy()

    if mode == "telea":
        # OpenCV edge-aware inpainting
        try:
            import cv2
        except Exception:
            raise ImportError("telea mode requires OpenCV (cv2). Install opencv-python.")
        # cv2.inpaint expects 8-bit single-channel mask: 255=inpaint, 0=keep
        m = hole.astype(np.uint8) * 255
        # Ensure uint8 BGR for OpenCV; support both uint8 and float
        if out.dtype != np.uint8:
            tmp = np.clip(out, 0, 255).astype(np.uint8)
        else:
            tmp = out
        # bgr = tmp[..., ::-1]  # RGB->BGR
        inpainted = cv2.inpaint(tmp, m, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        # out = inpainted[..., ::-1]  # BGR->RGB
        # If original was float, convert back to float range
        if img.dtype != np.uint8:
            out = out.astype(img.dtype)
        return out

    elif mode == "nearest":
        # Voronoi nearest-color fill using distance transform labels (OpenCV)
        try:
            import cv2
        except Exception:
            raise ImportError("nearest mode requires OpenCV (cv2). Install opencv-python.")

        # We want distance to the nearest VALID pixel.
        # cv2.distanceTransformWithLabels needs non-zero as "object" (valids),
        # so invert the mask: valid=1, holes=0
        valid = (~hole).astype(np.uint8)
        # Distance transform from holes to valids with labels
        dist, labels = cv2.distanceTransformWithLabels(
            valid, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL
        )
        # labels are 1..N for valid pixels, and 0 in holes where no label (shouldn’t happen if there’s any valid)
        # Recover coordinates of each valid pixel index
        ys, xs = np.mgrid[0:H, 0:W]
        # Flatten valid indices
        flat_idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
        # Map each pixel to its nearest valid pixel’s flat index:
        # Build lookup from label -> flat_index of that valid pixel.
        # A label corresponds to the position of the seed the DT grew from; OpenCV’s label equals 1+seed_index in raster scan.
        # We can reconstruct the seed map by setting each valid pixel’s label to its (1+flat_index), then running DT again.
        # More simply: create an array seeds where seeds[y,x] = 1+flat_index for valid pixels, else 0, and use labels as indices.

        seeds = np.zeros((H, W), dtype=np.int32)
        seeds[~hole] = flat_idx[~hole] + 1  # 1-based ids
        # Now labels[y,x] gives the 1-based id of nearest valid pixel; convert to flat index:
        nearest_flat = np.clip(labels - 1, 0, H * W - 1)

        # Pull colors from nearest valid pixel
        src = img.reshape(-1, img.shape[-1])
        out = img.copy()
        # Only fill hole pixels
        hole_flat = flat_idx[hole]
        out.reshape(-1, img.shape[-1])[hole_flat] = src[nearest_flat[hole]]
        return out

    elif mode == "box":
        out = img.astype(np.float32, copy=True)
        # Ensure we have a 2D mask for logic
        current_hole = hole.copy()
        known = (~current_hole).astype(np.float32)

        # Pre-mask the output so hole values don't pollute the sum
        out[current_hole] = 0.0

        for _ in range(radius):
            # 4-neighborhood sums of VALID pixels only
            sum_img = np.zeros_like(out)
            cnt = np.zeros_like(out)

            # Shift and accumulate ONLY known values
            # Up
            sum_img[1:] += out[:-1]
            cnt[1:] += known[:-1]
            # Down
            sum_img[:-1] += out[1:]
            cnt[:-1] += known[1:]
            # Left
            sum_img[:, 1:] += out[:, :-1]
            cnt[:, 1:] += known[:, :-1]
            # Right
            sum_img[:, :-1] += out[:, 1:]
            cnt[:, :-1] += known[:, 1:]

            # Identify pixels that are holes AND have at least one valid neighbor
            can_fill = current_hole & (cnt > 0)

            if not can_fill.any():
                break

            # Update values
            out[can_fill] = sum_img[can_fill] / cnt[can_fill]
            known[can_fill] = 1.0
            current_hole[can_fill] = False

            if not current_hole.any():
                break

        # Cast back to original dtype
        if img.dtype == np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = out.astype(img.dtype)
        return out[:, :]

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'telea', 'nearest', or 'box'.")


def infill_black_holes(tgt_rgb, mode="telea", radius=3):
    """
    Detect holes where tgt_rgb == 0 (black pixels) and infill them.
    """
    # hole mask = pixels where all channels == 0
    hole_mask = np.all(tgt_rgb == 0, axis=-1).astype(np.uint8)

    if not np.any(hole_mask):
        return tgt_rgb.copy(), hole_mask

    if mode == "telea":
        m = hole_mask * 255
        bgr = tgt_rgb[..., ::-1].astype(np.uint8)
        filled = cv2.inpaint(bgr, m, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        filled = filled[..., ::-1]  # back to RGB
    elif mode == "nearest":
        m = hole_mask * 255
        bgr = tgt_rgb[..., ::-1].astype(np.uint8)
        filled = cv2.inpaint(bgr, m, inpaintRadius=radius, flags=cv2.INPAINT_NS)
        filled = filled[..., ::-1]
    else:
        raise ValueError("mode must be 'telea' or 'nearest'")

    return filled, hole_mask
