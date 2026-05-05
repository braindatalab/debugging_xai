# Use from the root folder like: 
# python make_dataset.py --split-index 0 --position variable --include-masks --invert --scale neg_one_one
# Check the full list of args below to see which you need. 
# The default arguments are what are used in the main text of the paper


import os
import sys
import pickle
import random
import argparse
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image


# ----------------------------
# Utilities
# ----------------------------
SEEDS = [12031212, 1234, 5845389, 23423, 343495, 2024, 3842834, 23402304, 482347247, 1029237127]

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def rescale_zero_one(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def map_scale(arr01: np.ndarray, scale: str) -> np.ndarray:
    if scale == "zero_one":
        return arr01
    # "neg_one_one"
    return arr01 * 2.0 - 1.0

def map_unscale(arr_target: np.ndarray, scale: str) -> np.ndarray:
    if scale == "zero_one":
        return arr_target
    # convert [-1,1] -> [0,1]
    return (arr_target + 1.0) / 2.0

def load_image_as01(path: str, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize(size)
    return rescale_zero_one(np.array(im))

def invert01(arr01: np.ndarray) -> np.ndarray:
    inv = arr01.copy()
    inv[:, :, :3] = 1.0 - inv[:, :, :3]
    return inv

def preprocess_watermark_alpha(wm_path: str, out_width: int, thresh: float = 1/255) -> np.ndarray:
    """Return alpha in [0,1] where 0=background, 1=ink."""
    wm = Image.open(wm_path).convert("RGB")
    new_h = int(wm.size[1] * (out_width / wm.size[0]))
    wm = wm.resize((out_width, new_h))
    rgb01 = rescale_zero_one(np.array(wm))
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    alpha = 1.0 - (0.2989 * r + 0.5870 * g + 0.1140 * b)  # background≈0, ink≈1

    # trim empty rows/cols (JPEG white noise handled by thresh)
    row_keep = (alpha.max(axis=1) > thresh)
    col_keep = (alpha.max(axis=0) > thresh)
    alpha = alpha[row_keep][:, col_keep]
    alpha[alpha < thresh] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return alpha

def _wm_alpha_fullwidth(wm_path: str, out_width: int, thresh: float = 5/255) -> np.ndarray:
    """Resize to image width, compute alpha in [0,1]; do NOT trim (keeps margins)."""
    wm = Image.open(wm_path).convert("RGB")
    new_h = int(wm.size[1] * (out_width / wm.size[0]))
    wm = wm.resize((out_width, new_h))
    rgb = np.asarray(wm, dtype=np.float32)
    rgb = (rgb - rgb.min()) / max(1e-8, (rgb.max() - rgb.min()))  # to [0,1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    alpha = 1.0 - (0.2989 * r + 0.5870 * g + 0.1140 * b)          # white≈0, ink≈1
    alpha[alpha < thresh] = 0.0
    return alpha.astype(np.float32)

def _wm_alpha_trimmed(wm_path: str, out_width: int, thresh: float = 5/255) -> np.ndarray:
    """Resize to image width, compute alpha, then trim empty rows/cols (for variable placement)."""
    alpha = _wm_alpha_fullwidth(wm_path, out_width, thresh)
    row_keep = (alpha.max(axis=1) > thresh)
    col_keep = (alpha.max(axis=0) > thresh)
    if row_keep.any(): alpha = alpha[row_keep]
    if col_keep.any(): alpha = alpha[:, col_keep]
    return alpha

def compose_watermark01(
    bg01: np.ndarray,
    wm_path: str,
    intensity: float,
    white_on: bool,
    position_mode: str,
    alpha_thresh: float = 5/255,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alpha blend only the ink: out = (1-a)*bg + a*color, a = alpha*intensity.
    Fixed: resize to full width, keep margins, place at (0,0) like original.
    Variable: trim and place at a random (y,x).
    """
    H, W, _ = bg01.shape

    if position_mode == "fixed":
        alpha = _wm_alpha_fullwidth(wm_path, out_width=W, thresh=alpha_thresh)
        y, x = 0, 0
    else:
        alpha = _wm_alpha_trimmed(wm_path, out_width=W, thresh=alpha_thresh)
        h, w = alpha.shape
        y = np.random.randint(0, max(1, H - h + 1))
        x = np.random.randint(0, max(1, W - w + 1))

    # paste alpha into full canvas (crop if needed)
    alpha_full = np.zeros((H, W), dtype=np.float32)
    h, w = alpha.shape
    y2, x2 = min(H, y + h), min(W, x + w)
    hh, ww = y2 - y, x2 - x
    if hh > 0 and ww > 0:
        alpha_full[y:y2, x:x2] = alpha[:hh, :ww]

    a = np.clip(alpha_full * float(intensity), 0.0, 1.0)[..., None]
    color = 1.0 if white_on else 0.0
    out01 = bg01 * (1.0 - a) + color * a

    mask01 = (alpha_full > alpha_thresh).astype(np.float32)
    return out01.astype(np.float32), mask01


def random_position(wm_shape: tuple[int, int], image_shape: tuple[int, int]) -> tuple[int, int]:
    H, W = image_shape
    h, w = wm_shape
    y = np.random.randint(0, max(1, H - h + 1))
    x = np.random.randint(0, max(1, W - w + 1))
    return y, x


# ----------------------------
# Dataset writer
# ----------------------------
def build_split_indices(N: int, train_p=0.7, val_p=0.15):
    all_idx = np.arange(N)
    train_idx = np.random.choice(all_idx, size=int(N * train_p), replace=False)
    rem = np.setdiff1d(all_idx, train_idx)
    val_idx = np.random.choice(rem, size=int(N * val_p), replace=False)
    test_idx = np.setdiff1d(rem, val_idx)
    return train_idx, val_idx, test_idx

def save_pickle(path: str, payload: list):
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def make_split_dataset(
    cat_files: list[str],
    dog_files: list[str],
    wm_path: str,
    image_size: tuple[int, int],
    intensity: float,
    wm_prev_cat: float,
    wm_prev_dog: float,
    out_path: str,
    white_on: bool,
    scale: str,
    position_mode: str,
    include_masks: bool,
    invert: bool,
):
    H, W = image_size
    total = len(cat_files) + len(dog_files)
    data = np.zeros((total, H, W, 3), dtype=np.float32)
    labels = np.zeros((total, 1), dtype=np.float32)
    labels[len(cat_files):] = 1.0
    watermark_inds = []

    if include_masks:
        masks = np.zeros((total, H, W), dtype=np.float32)
    else:
        masks = None

    cat_wm = set(np.random.choice(cat_files, size=int(len(cat_files) * wm_prev_cat), replace=False))
    dog_wm = set(np.random.choice(dog_files, size=int(len(dog_files) * wm_prev_dog), replace=False))

    idx = 0
    for files, is_dog, pool in [(cat_files, 0, cat_wm), (dog_files, 1, dog_wm)]:
        for p in files:
            bg01 = load_image_as01(p, image_size)

            if p in pool:
                out01, m01 = compose_watermark01(
                    bg01, wm_path, intensity=intensity, white_on=white_on, position_mode=position_mode
                )
                watermark_inds.append(idx)
                if include_masks:
                    masks[idx] = m01
            else:
                out01 = bg01
                if include_masks:
                    masks[idx] = 0.0

            if invert:
                out01 = invert01(out01)

            data[idx] = map_scale(out01, scale)
            idx += 1

    print(f"[{out_path}] #with_wm: {len(watermark_inds)}  #without_wm: {total - len(watermark_inds)}")

    if include_masks:
        save_pickle(out_path, [data, labels, watermark_inds, masks])
    else:
        save_pickle(out_path, [data, labels, watermark_inds])


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate dog/cat datasets with optional watermarks.")
    parser.add_argument("--split-index", type=int, default=0, help="Index into predefined SEEDS.")
    parser.add_argument("--cats-dir", type=str, default="./images/cat", help="Path to cat images.")
    parser.add_argument("--dogs-dir", type=str, default="./images/dog", help="Path to dog images.")
    parser.add_argument("--watermark", type=str, default="./watermark banner.jpg", help="Path to watermark image (jpg).")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--intensity", type=float, default=0.8, help="Watermark intensity in [0,1].")
    parser.add_argument("--white-on", type=int, default=1, choices=[0, 1], help="1 if watermark is white-on, else 0.")
    parser.add_argument("--N", type=int, default=4800, help="Images per class.")
    parser.add_argument("--scale", choices=["zero_one", "neg_one_one"], default="zero_one",
                        help="Output scaling. zero_one=[0,1] (default), neg_one_one=[-1,1].")
    parser.add_argument("--position", choices=["fixed", "variable"], default="fixed",
                        help="Fixed top-left placement (resized to width) vs random translation.")
    parser.add_argument("--include-masks", action="store_false",
                        help="If set and --position variable, include masks in the pickle (default off).")
    parser.add_argument("--invert", action="store_true", help="Invert RGB channels (1 - x) in [0,1] space before final scaling.")
    parser.add_argument("--outdir", type=str, default="./artifacts", help="Output directory.")

    args = parser.parse_args()

    if args.split_index < 0 or args.split_index >= len(SEEDS):
        raise ValueError(f"--split-index must be in [0, {len(SEEDS)-1}]")

    set_seed(SEEDS[args.split_index])

    cat_paths = sorted(glob(os.path.join(args.cats_dir, "*")))
    dog_paths = sorted(glob(os.path.join(args.dogs_dir, "*")))
    print("total dogs and cats:", len(dog_paths), len(cat_paths))

    if len(cat_paths) < args.N or len(dog_paths) < args.N:
        raise ValueError("Not enough images to sample N per class for all splits.")

    # Build consistent splits by sampling indices (per class)
    idx_train, idx_val, idx_test = build_split_indices(args.N)

    cat_train = list(np.array(cat_paths)[:args.N][idx_train])
    cat_val   = list(np.array(cat_paths)[:args.N][idx_val])
    cat_test  = list(np.array(cat_paths)[:args.N][idx_test])

    dog_train = list(np.array(dog_paths)[:args.N][idx_train])
    dog_val   = list(np.array(dog_paths)[:args.N][idx_val])
    dog_test  = list(np.array(dog_paths)[:args.N][idx_test])

    # Defaults mirror original behavior:
    # - Generate four settings per split: suppressor, confounder, no_watermark, all_watermark
    # - Default scaling [0,1]
    # - Default position fixed
    # - No masks unless explicitly requested and position=variable
    H, W = args.image_size
    rescaled_suffix = "" if args.scale == "zero_one" else "_rescaled"
    pos_suffix = "" if args.position == "fixed" else "_variablepos"
    inv_suffix = "_inverted" if args.invert else ""

    def out(base, subset):
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        return os.path.join(
            args.outdir,
            f"split_{args.split_index}_{base}_{subset}{rescaled_suffix}{pos_suffix}{inv_suffix}.pkl"
        )

    # suppressor: 50/50 in both classes
    base = "suppressor"
    make_split_dataset(cat_train, dog_train, args.watermark, (H, W), args.intensity, 0.5, 0.5,
                       out(base, "train"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)
    make_split_dataset(cat_val, dog_val, args.watermark, (H, W), args.intensity, 0.5, 0.5,
                       out(base, "val"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)
    make_split_dataset(cat_test, dog_test, args.watermark, (H, W), args.intensity, 0.5, 0.5,
                       out(base, "test"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)

    # confounder: 20% cat, 80% dog
    base = "confounder"
    make_split_dataset(cat_train, dog_train, args.watermark, (H, W), args.intensity, 0.2, 0.8,
                       out(base, "train"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)
    make_split_dataset(cat_val, dog_val, args.watermark, (H, W), args.intensity, 0.2, 0.8,
                       out(base, "val"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)
    make_split_dataset(cat_test, dog_test, args.watermark, (H, W), args.intensity, 0.2, 0.8,
                       out(base, "test"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)

    # no watermark
    base = "no_watermark"
    make_split_dataset(cat_train, dog_train, args.watermark, (H, W), args.intensity, 0.0, 0.0,
                       out(base, "train"), args.white_on == 1, args.scale, args.position,
                       include_masks=False, invert=args.invert)
    make_split_dataset(cat_val, dog_val, args.watermark, (H, W), args.intensity, 0.0, 0.0,
                       out(base, "val"), args.white_on == 1, args.scale, args.position,
                       include_masks=False, invert=args.invert)
    make_split_dataset(cat_test, dog_test, args.watermark, (H, W), args.intensity, 0.0, 0.0,
                       out(base, "test"), args.white_on == 1, args.scale, args.position,
                       include_masks=False, invert=args.invert)

    # all watermark (test only)
    base = "all_watermark"
    make_split_dataset(cat_test, dog_test, args.watermark, (H, W), args.intensity, 1.0, 1.0,
                       out(base, "test"), args.white_on == 1, args.scale, args.position,
                       include_masks=(args.include_masks and args.position == "variable"),
                       invert=args.invert)

if __name__ == "__main__":
    main()
