import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.util import view_as_windows

from sbef import SparseBayesExpander


TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
R = 2
PATCH_SIZE = 11
A_ALPHA0 = 20.0
CUBIC_KERNEL_A = -0.5

RGB_TO_YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
], dtype=np.float64)
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ)


def rgb_to_yiq(img):
    flat = img.reshape(-1, 3)
    return (flat @ RGB_TO_YIQ.T).reshape(img.shape)


def yiq_to_rgb(img):
    flat = img.reshape(-1, 3)
    return (flat @ YIQ_TO_RGB.T).reshape(img.shape)


def crop_to_factor(img, r):
    h, w = img.shape[:2]
    return img[:(h // r) * r, :(w // r) * r]


def cubic_kernel(x, a=CUBIC_KERNEL_A):
    """Keys cubic convolution kernel used for the anti-aliasing blur."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    weights = np.zeros_like(x)

    mask1 = x <= 1
    weights[mask1] = (
        (a + 2) * x[mask1] ** 3
        - (a + 3) * x[mask1] ** 2
        + 1
    )

    mask2 = (x > 1) & (x < 2)
    weights[mask2] = (
        a * x[mask2] ** 3
        - 5 * a * x[mask2] ** 2
        + 8 * a * x[mask2]
        - 4 * a
    )

    return weights


def cubic_antialias_filter(r):
    """Return a scaled cubic low-pass kernel for decimation by factor r."""
    if r < 1:
        raise ValueError('r must be a positive integer')
    radius = 2 * int(r)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = cubic_kernel(offsets / float(r)) / float(r)
    kernel_sum = np.sum(kernel)
    if not np.isfinite(kernel_sum) or abs(kernel_sum) < 1e-12:
        raise ValueError(f'invalid cubic anti-aliasing kernel for r={r}')
    return kernel / kernel_sum


def make_low_resolution(img, r):
    """Create the low-resolution image by cubic blur followed by subsampling.

    The paper states this step as cubic-kernel anti-aliasing followed by
    subsampling by the expansion factor.  This function keeps those two steps
    explicit instead of using a combined resize call.
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = cubic_antialias_filter(r)
    blurred = ndimage.convolve1d(img, kernel, axis=0, mode='nearest')
    blurred = ndimage.convolve1d(blurred, kernel, axis=1, mode='nearest')
    return blurred[::r, ::r, ...].astype(np.float64)


def cubic_expand(img, out_shape):
    img = np.asarray(img, dtype=np.float64)

    if img.ndim == 2:
        in_h, in_w = img.shape
        out_h, out_w = out_shape
        r_h = out_h / in_h
        r_w = out_w / in_w

        yy = np.arange(out_h, dtype=np.float64) / r_h
        xx = np.arange(out_w, dtype=np.float64) / r_w
        coords = np.meshgrid(yy, xx, indexing='ij')

        return ndimage.map_coordinates(
            img,
            coords,
            order=3,
            mode='nearest'
        ).astype(np.float64)

    channels = [
        cubic_expand(img[:, :, c], out_shape[:2])
        for c in range(img.shape[2])
    ]
    return np.stack(channels, axis=2)


def extract_patches(channel, r=2, m=11, train_mode=False):
    """Return one-channel patch matrices Y and X.

    High-resolution patches are non-overlapping r x r blocks.  In training mode
    boundary low-resolution patches are discarded.  In test/application mode the
    low-resolution image is extended by pixel replication before extracting
    patches.
    """
    channel = crop_to_factor(np.asarray(channel, dtype=np.float64), r)
    low = make_low_resolution(channel, r)
    pad = m // 2
    low_padded = np.pad(low, pad, mode='edge')
    all_patches = view_as_windows(low_padded, (m, m))
    h_low, w_low = low.shape

    if train_mode:
        if h_low <= 2 * pad or w_low <= 2 * pad:
            raise ValueError(f'image is too small for m={m}; low-res size is {low.shape}')
        centers_i = range(pad, h_low - pad)
        centers_j = range(pad, w_low - pad)
    else:
        centers_i = range(h_low)
        centers_j = range(w_low)

    y_cols = []
    x_cols = []
    for i in centers_i:
        for j in centers_j:
            y_cols.append(all_patches[i, j].reshape(-1))
            x_cols.append(channel[i * r:(i + 1) * r, j * r:(j + 1) * r].reshape(-1))

    return np.asarray(y_cols, dtype=np.float64).T, np.asarray(x_cols, dtype=np.float64).T


def expand_channel(model, channel, r=2, m=11):
    channel = crop_to_factor(np.asarray(channel, dtype=np.float64), r)
    low = make_low_resolution(channel, r)
    pad = m // 2
    low_padded = np.pad(low, pad, mode='edge')
    h_low, w_low = low.shape
    out = np.zeros((h_low * r, w_low * r), dtype=np.float64)

    for i in range(h_low):
        for j in range(w_low):
            patch = low_padded[i:i + m, j:j + m].reshape(-1)
            out[i * r:(i + 1) * r, j * r:(j + 1) * r] = model.transform_patch(patch).reshape((r, r))

    return out[:channel.shape[0], :channel.shape[1]], low


def load_images_from_dir(directory):
    allowed = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    imgs = []
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            im = io.imread(str(path))
        except Exception as exc:
            print(f'Warning: failed to read image {path}: {exc}')
            continue
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=2)
        if im.ndim == 3 and im.shape[2] > 3:
            im = im[:, :, :3]
        im = im.astype(np.float64)
        if im.max() > 1.0:
            im /= 255.0
        imgs.append(np.clip(im, 0.0, 1.0))
    return imgs


def channel_training_data(images, channel_fn, r, m):
    ys = []
    xs = []
    for img in images:
        channel = channel_fn(crop_to_factor(img, r))
        y, x = extract_patches(channel, r=r, m=m, train_mode=True)
        ys.append(y)
        xs.append(x)
    return np.concatenate(ys, axis=1), np.concatenate(xs, axis=1)


def fit_one_channel(y, x, args):
    model = SparseBayesExpander(
        D=x.shape[0], Q=y.shape[0], a_alpha0=args.a_alpha0,
        b_alpha0=args.b_alpha0, a_beta0=args.a_beta0, b_beta0=args.b_beta0,
        alpha_threshold=np.exp(20), max_iter=args.max_iter, tol=args.tol,
        verbose=args.verbose, symmetry=args.symmetry,
    )
    return model.fit(y, x)


def fit_models(train_imgs, mode, args):
    models = []
    if mode == 'rgb':
        for ch in range(3):
            y, x = channel_training_data(train_imgs, lambda img, c=ch: img[:, :, c], args.r, args.m)
            models.append(fit_one_channel(y, x, args))
    elif mode == 'yiq-luma':
        y, x = channel_training_data(train_imgs, lambda img: rgb_to_yiq(img)[:, :, 0], args.r, args.m)
        models.append(fit_one_channel(y, x, args))
    elif mode == 'yiq-all':
        for ch in range(3):
            y, x = channel_training_data(train_imgs, lambda img, c=ch: rgb_to_yiq(img)[:, :, c], args.r, args.m)
            models.append(fit_one_channel(y, x, args))
    else:
        raise ValueError(f'unknown mode: {mode}')
    return models


def expand_image(models, img, mode, r, m):
    img = crop_to_factor(img, r)
    if mode == 'rgb':
        rec = np.zeros_like(img)
        lows = []
        for ch in range(3):
            rec[:, :, ch], low = expand_channel(models[ch], img[:, :, ch], r=r, m=m)
            lows.append(low)
        return np.clip(rec, 0.0, 1.0), np.stack(lows, axis=2)

    yiq = rgb_to_yiq(img)
    if mode == 'yiq-luma':
        out_y, low_y = expand_channel(models[0], yiq[:, :, 0], r=r, m=m)
        low_i = make_low_resolution(yiq[:, :, 1], r)
        low_q = make_low_resolution(yiq[:, :, 2], r)
        out_i = cubic_expand(low_i, img.shape[:2])
        out_q = cubic_expand(low_q, img.shape[:2])
        rec_yiq = np.stack([out_y, out_i, out_q], axis=2)
        low_rgb = np.clip(yiq_to_rgb(np.stack([low_y, low_i, low_q], axis=2)), 0.0, 1.0)
        return np.clip(yiq_to_rgb(rec_yiq), 0.0, 1.0), low_rgb

    if mode == 'yiq-all':
        rec_yiq = np.zeros_like(yiq)
        lows = []
        for ch in range(3):
            rec_yiq[:, :, ch], low = expand_channel(models[ch], yiq[:, :, ch], r=r, m=m)
            lows.append(low)
        low_rgb = np.clip(yiq_to_rgb(np.stack(lows, axis=2)), 0.0, 1.0)
        return np.clip(yiq_to_rgb(rec_yiq), 0.0, 1.0), low_rgb

    raise ValueError(f'unknown mode: {mode}')


def cubic_baseline(img, r):
    img = crop_to_factor(img, r)
    low = make_low_resolution(img, r)
    return np.clip(cubic_expand(low, img.shape), 0.0, 1.0), np.clip(low, 0.0, 1.0)


def evaluate_models(models, test_imgs, mode, r, m):
    learned_scores = []
    cubic_scores = []
    first = None
    for img in test_imgs:
        img = crop_to_factor(img, r)
        rec, low = expand_image(models, img, mode, r, m)
        cubic, cubic_low = cubic_baseline(img, r)
        learned_scores.append(sk_psnr(img, rec, data_range=1.0))
        cubic_scores.append(sk_psnr(img, cubic, data_range=1.0))
        if first is None:
            first = (img, low, rec, cubic, cubic_low)
    return learned_scores, cubic_scores, first


def save_support_maps(models, mode, m, out_dir):
    names = ['R', 'G', 'B'] if mode == 'rgb' else (['Y'] if mode == 'yiq-luma' else ['Y', 'I', 'Q'])
    for name, model in zip(names, models):
        support_map = model.support_union_mask().reshape((m, m)).astype(float)
        plt.figure(figsize=(4, 4))
        plt.imshow(support_map, cmap='gray', vmin=0, vmax=1)
        plt.title(f'ARD support ({name}), size={model.support_size()}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'support_{name}.png', dpi=150)
        plt.close()


def save_demo_outputs(first, learned_scores, cubic_scores, mode, out_dir):
    img, low, rec, cubic, _ = first
    plt.imsave(out_dir / 'original.png', img)
    plt.imsave(out_dir / 'low_res.png', np.clip(low, 0.0, 1.0))
    plt.imsave(out_dir / 'cubic.png', np.clip(cubic, 0.0, 1.0))
    plt.imsave(out_dir / 'learned.png', np.clip(rec, 0.0, 1.0))

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[1].imshow(np.clip(low, 0.0, 1.0))
    axes[1].set_title('Low-res')
    axes[2].imshow(np.clip(cubic, 0.0, 1.0))
    axes[2].set_title(f'Cubic {cubic_scores[0]:.2f} dB')
    axes[3].imshow(np.clip(rec, 0.0, 1.0))
    axes[3].set_title(f'{mode} {learned_scores[0]:.2f} dB')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / 'comparison.png', dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Bayesian image expansion demo')
    parser.add_argument('--train-dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--test-dir', type=str, default=TEST_DIR)
    parser.add_argument('--mode', choices=['rgb', 'yiq-luma', 'yiq-all'], default='yiq-all')
    parser.add_argument('--r', type=int, default=R)
    parser.add_argument('--m', type=int, default=PATCH_SIZE)
    parser.add_argument('--a-alpha0', type=float, default=A_ALPHA0)
    parser.add_argument('--b-alpha0', type=float, default=2e-8)
    parser.add_argument('--a-beta0', type=float, default=1e-6)
    parser.add_argument('--b-beta0', type=float, default=1e-6)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--symmetry', choices=['none', 'h', 'v', 'hv'], default='hv')
    parser.add_argument('--output-dir', type=str, default='demo_outputs')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise SystemExit(f'training directory does not exist: {train_dir}')
    if not test_dir.exists():
        raise SystemExit(f'test directory does not exist: {test_dir}')

    train_imgs = load_images_from_dir(train_dir)
    test_imgs = load_images_from_dir(test_dir)
    if not train_imgs:
        raise SystemExit(f'no training images found in {train_dir}')
    if not test_imgs:
        raise SystemExit(f'no test images found in {test_dir}')

    print(f'Loaded {len(train_imgs)} training images and {len(test_imgs)} test images')
    print(f'Training mode={args.mode}, r={args.r}, m={args.m}, a_alpha0={args.a_alpha0}')

    models = fit_models(train_imgs, args.mode, args)
    learned_scores, cubic_scores, first = evaluate_models(models, test_imgs, args.mode, args.r, args.m)

    print(f'Mean PSNR learned: {np.mean(learned_scores):.4f} dB')
    print(f'Mean PSNR cubic baseline: {np.mean(cubic_scores):.4f} dB')
    print('Support sizes:', [model.support_size() for model in models])

    save_support_maps(models, args.mode, args.m, out_dir)
    save_demo_outputs(first, learned_scores, cubic_scores, args.mode, out_dir)
    print(f'Saved outputs to {out_dir}')


if __name__ == '__main__':
    main()
