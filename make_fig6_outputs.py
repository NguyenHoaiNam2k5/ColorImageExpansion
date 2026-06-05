import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from demo import fit_models, load_images_from_dir, make_low_resolution
from inference import cubic_baseline_from_low, expand_lr_image


MODE_TO_FILENAME = {
    'cubic': 'cubic.png',
    'yiq-luma': 'y_plus_cubic.png',
    'rgb': 'rgb.png',
    'yiq-all': 'yiq.png',
}

MODE_TO_LABEL = {
    'cubic': 'Cubic',
    'yiq-luma': 'Y + cubic',
    'rgb': 'RGB',
    'yiq-all': 'YIQ',
}


def load_rgb_image(path):
    """Load one image as float64 RGB in [0, 1]."""
    img = io.imread(str(path))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f'expected a grayscale or RGB image, got shape {img.shape}')

    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def crop_image(img, x, y, size):
    """Crop a square patch using image coordinates x=column, y=row."""
    h, w = img.shape[:2]
    if size <= 0:
        raise ValueError('crop size must be positive')
    if x < 0 or y < 0 or x + size > w or y + size > h:
        raise ValueError(
            f'crop box {(x, y, x + size, y + size)} is outside image size {(w, h)}'
        )
    return img[y:y + size, x:x + size, :]


def nearest_expand(low, r, out_shape):
    """Display a low-resolution image at the high-resolution panel size."""
    up = np.repeat(np.repeat(low, r, axis=0), r, axis=1)
    return up[:out_shape[0], :out_shape[1], :]


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(img, 0.0, 1.0))


def make_model_args(args):
    return SimpleNamespace(
        r=args.r,
        m=args.m,
        a_alpha0=args.a_alpha0,
        b_alpha0=args.b_alpha0,
        a_beta0=args.a_beta0,
        b_beta0=args.b_beta0,
        max_iter=args.max_iter,
        tol=args.tol,
        symmetry=args.symmetry,
        verbose=args.verbose,
    )


def train_all_modes(train_imgs, args):
    model_args = make_model_args(args)
    trained = {}
    for mode in ('yiq-luma', 'rgb', 'yiq-all'):
        print(f'Training {MODE_TO_LABEL[mode]} with a_alpha0={args.a_alpha0:g}')
        trained[mode] = fit_models(train_imgs, mode, model_args)
    return trained


def save_metrics(out_dir, original, outputs):
    rows = []
    for mode, image in outputs.items():
        rows.append({
            'method': MODE_TO_LABEL[mode],
            'filename': MODE_TO_FILENAME[mode],
            'psnr_db': f'{sk_psnr(original, image, data_range=1.0):.6f}',
        })

    metrics_path = out_dir / 'metrics.csv'
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'filename', 'psnr_db'])
        writer.writeheader()
        writer.writerows(rows)

    return rows, metrics_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train the three color-expansion schemes at a_alpha0=20 and save the '
            'six images used for a Fig. 6-style qualitative comparison.'
        )
    )
    parser.add_argument('image_path', type=str, help='Path to the original high-resolution image')
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--output-dir', type=str, default='fig6_outputs')
    parser.add_argument('--x', type=int, default=230, help='Left coordinate of the crop')
    parser.add_argument('--y', type=int, default=220, help='Top coordinate of the crop')
    parser.add_argument('--crop-size', type=int, default=64, help='Square crop size in HR pixels')
    parser.add_argument('--r', type=int, default=2, help='Expansion factor')
    parser.add_argument('--m', type=int, default=11, help='Low-resolution patch size')
    parser.add_argument('--a-alpha0', type=float, default=20.0)
    parser.add_argument('--b-alpha0', type=float, default=2e-8)
    parser.add_argument('--a-beta0', type=float, default=1e-6)
    parser.add_argument('--b-beta0', type=float, default=1e-6)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--symmetry', choices=['none', 'h', 'v', 'hv'], default='hv')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.crop_size % args.r != 0:
        raise SystemExit('crop-size must be divisible by r so the LR image has integer dimensions')

    train_imgs = load_images_from_dir(args.train_dir)
    if not train_imgs:
        raise SystemExit(f'no training images found in {args.train_dir}')

    original_full = load_rgb_image(Path(args.image_path))
    original = crop_image(original_full, args.x, args.y, args.crop_size)
    low = make_low_resolution(original, args.r)
    low_display = nearest_expand(low, args.r, original.shape)

    models_by_mode = train_all_modes(train_imgs, args)

    outputs = {
        'cubic': cubic_baseline_from_low(low, r=args.r, out_shape=original.shape),
        'yiq-luma': expand_lr_image(
            models_by_mode['yiq-luma'], low, mode='yiq-luma',
            r=args.r, m=args.m, out_shape=original.shape,
        ),
        'rgb': expand_lr_image(
            models_by_mode['rgb'], low, mode='rgb',
            r=args.r, m=args.m, out_shape=original.shape,
        ),
        'yiq-all': expand_lr_image(
            models_by_mode['yiq-all'], low, mode='yiq-all',
            r=args.r, m=args.m, out_shape=original.shape,
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(out_dir / 'original.png', original)
    save_rgb(out_dir / 'low_resolution.png', low_display)
    for mode, image in outputs.items():
        save_rgb(out_dir / MODE_TO_FILENAME[mode], image)

    rows, metrics_path = save_metrics(out_dir, original, outputs)

    print(f'Saved six images to {out_dir}:')
    print('  original.png')
    print('  low_resolution.png')
    for mode in ('cubic', 'yiq-luma', 'rgb', 'yiq-all'):
        print(f'  {MODE_TO_FILENAME[mode]}')

    print(f'Saved metrics to {metrics_path}:')
    for row in rows:
        print(f"  {row['method']}: {row['psnr_db']} dB")


if __name__ == '__main__':
    main()
