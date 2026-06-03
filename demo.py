import numpy as np
from skimage import data, transform
from skimage.util import view_as_windows
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from sbef import SparseBayesExpander
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from skimage import io

# --------------------
# CONFIG: edit these values directly to set dataset and patch params
# Set TRAIN_DIR/TEST_DIR to a string path to your folders, or None to use defaults
TRAIN_DIR = r"data\train"
TEST_DIR = r"data\test"
R = 2   # magnification factor (r)
M = 11  # low-resolution patch size (m)
# --------------------


def extract_patches(img, r=2, m=11, train_mode=False):
    """Extract low/high patches according to the paper's experiment.

    - High-res patches: non-overlapping r x r blocks taken from the image grid.
    - Low-res: blur with cubic kernel and subsample by r (implemented with
      transform.resize(order=3, anti_aliasing=True, preserve_range=True)).
    - For training (train_mode=True): discard low-res patches that rely on
      padding at the image boundary (paper: discard boundary patches).
    - For test (train_mode=False): extend the low-res image by pixel
      replication (pad with mode='edge') before extracting overlapping patches.
    Returns Y (Q x N) and X (D x N) where Q = m*m and D = r*r.
    """
    H, W = img.shape
    # crop to a multiple of r so high-res blocks are exact
    Hr = (H // r) * r
    Wr = (W // r) * r
    img = img[:Hr, :Wr]

    # create low-resolution image by cubic anti-aliased resize
    low = transform.resize(img, (Hr // r, Wr // r), order=3, anti_aliasing=True, preserve_range=True)

    pad = m // 2
    # For test-mode we must extend low by pixel replication (edge) as the paper
    # requires. For training we still pad to allow extracting patches, but we
    # will discard boundary patches afterwards.
    low_padded = np.pad(low, pad, mode='edge')
    patches = view_as_windows(low_padded, (m, m))  # shape (H', W', m, m)
    Hs, Ws = patches.shape[:2]
    Q = m * m

    if train_mode:
        # discard patches whose centers were within 'pad' pixels of the low-res border
        Hs_valid = Hs - 2 * pad
        Ws_valid = Ws - 2 * pad
        if Hs_valid <= 0 or Ws_valid <= 0:
            Y = patches.reshape(-1, Q).T
        else:
            Y = patches[pad:pad+Hs_valid, pad:pad+Ws_valid, :, :].reshape(-1, Q).T
            Hs, Ws = Hs_valid, Ws_valid
    else:
        Y = patches.reshape(-1, Q).T

    # corresponding high-res patches: for each low patch take r x r block
    X_list = []
    for i in range(Hs):
        for j in range(Ws):
            hi = i * r
            hj = j * r
            xr = img[hi:hi+r, hj:hj+r].reshape(-1)
            X_list.append(xr)
    X = np.array(X_list).T  # D x N
    return Y, X


def main():
    parser = argparse.ArgumentParser(description='Demo: Sparse Bayesian Image Expansion')
    parser.add_argument('--train-dir', type=str, default=None, help='directory with training images')
    parser.add_argument('--test-dir', type=str, default=None, help='directory with test images')
    parser.add_argument('--r', type=int, default=None, help='magnification factor (overrides config R)')
    parser.add_argument('--m', type=int, default=None, help='low-res patch size (overrides config M)')
    parser.add_argument('--mode', type=str, default='rgb', choices=['rgb', 'yiq-luma', 'yiq-all'], help='expansion mode: rgb (learn each RGB), yiq-luma (learn Y only, cubic for I/Q), yiq-all (learn Y,I,Q)')
    args = parser.parse_args()

    # Use CLI args if provided, otherwise fall back to top-level CONFIG
    r = args.r if args.r is not None else R
    m = args.m if args.m is not None else M
    mode = args.mode
    img = None
    # r and m potentially set by command-line
    # r = magnification factor, m = low-res patch size
    # If user provided a training directory, we'll build train/test sets from files.
    train_dir = None if args.train_dir is None else Path(args.train_dir)
    # if CLI not provided, use config strings (which may be None)
    if train_dir is None and TRAIN_DIR:
        train_dir = Path(TRAIN_DIR)

    test_dir = None if args.test_dir is None else Path(args.test_dir)
    if test_dir is None and TEST_DIR:
        test_dir = Path(TEST_DIR)

    if train_dir is None and test_dir is None:
        parser.error('Please provide at least --train-dir or --test-dir (no default astronaut fallback).')

    reconstructed = np.zeros_like(img)
    learned_kernels = []

    out_dir = Path.cwd() / 'demo_outputs'
    out_dir.mkdir(exist_ok=True)

    # color conversion helpers for YIQ
    M_rgb2yiq = np.array([[0.299,  0.587,  0.114],
                          [0.596, -0.274, -0.322],
                          [0.211, -0.523,  0.312]])
    M_yiq2rgb = np.linalg.inv(M_rgb2yiq)

    def rgb_to_yiq(img):
        shp = img.shape
        flat = img.reshape(-1, 3)
        yiq = flat @ M_rgb2yiq.T
        return yiq.reshape(shp)

    def yiq_to_rgb(img):
        shp = img.shape
        flat = img.reshape(-1, 3)
        rgb = flat @ M_yiq2rgb.T
        return rgb.reshape(shp)

    def load_images_from_dir(d):
        # load common image types (recursive); accept .tif and .tiff, case-insensitive
        allowed = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        imgs = []
        # walk recursively to find files with allowed suffixes
        for p in sorted(Path(d).rglob('*')):
            if p.is_file() and p.suffix.lower() in allowed:
                try:
                    im = io.imread(str(p))
                except Exception as e:
                    print(f'Warning: failed to read image {p!s}: {e}')
                    continue
                if im.ndim == 2:
                    im = np.stack([im, im, im], axis=2)
                im = im.astype(np.float32) / 255.0
                imgs.append(im)
        return imgs

    # Use the standard single-channel PSNR (skimage convention):
    # PSNR = 10 * log10(1 / MSE) where pixel range is [0,1].

    # If train/test dirs provided, prepare lists
    train_imgs = []
    test_imgs = []
    if train_dir is not None:
        if train_dir.exists():
            train_imgs = load_images_from_dir(train_dir)
            if not train_imgs:
                print(f'Warning: no training images found in {train_dir!s} (patterns: png,jpg,jpeg,tif,bmp)')
            else:
                print(f'Loaded {len(train_imgs)} training images from {train_dir!s}')
        else:
            print(f'Warning: TRAIN_DIR provided but path does not exist: {train_dir!s}')
    if test_dir is not None:
        if test_dir.exists():
            test_imgs = load_images_from_dir(test_dir)
            if not test_imgs:
                print(f'Warning: no test images found in {test_dir!s} (patterns: png,jpg,jpeg,tif,bmp)')
            else:
                print(f'Loaded {len(test_imgs)} test images from {test_dir!s}')
        else:
            print(f'Warning: TEST_DIR provided but path does not exist: {test_dir!s}')

    # Training data: either single default image (astronaut crop) or images from train_dir
    if train_imgs:
        models = []
        if mode == 'rgb':
            # accumulate patches for each channel across all training images
            channel_train_data = [[], [], []]
            for tr in train_imgs:
                H, W = tr.shape[:2]
                Hc = (H // r) * r
                Wc = (W // r) * r
                tr = tr[:Hc, :Wc]
                for ch in range(3):
                    Yt, Xt = extract_patches(tr[:, :, ch], r=r, m=m)
                    channel_train_data[ch].append((Yt, Xt))
            for ch in range(3):
                Ys = [t[0] for t in channel_train_data[ch]]
                Xs = [t[1] for t in channel_train_data[ch]]
                if not Ys:
                    raise SystemExit('No training patches found for channel; provide training images.')
                Y_all = np.concatenate(Ys, axis=1)
                X_all = np.concatenate(Xs, axis=1)
                D = X_all.shape[0]
                Q = Y_all.shape[0]
                model = SparseBayesExpander(D, Q, a_alpha0=20.0, max_iter=200, verbose=True)
                model.fit(Y_all, X_all)
                learned_kernels.append(model.M.copy())
                models.append(model)
        elif mode == 'yiq-luma':
            # convert training images to YIQ and only train on Y channel
            Y_channel_patches = []
            X_channel_patches = []
            for tr in train_imgs:
                H, W = tr.shape[:2]
                Hc = (H // r) * r
                Wc = (W // r) * r
                tr = tr[:Hc, :Wc]
                tri = rgb_to_yiq(tr)
                # use luminance channel (0)
                Yt, Xt = extract_patches(tri[:, :, 0], r=r, m=m)
                Y_channel_patches.append((Yt, Xt))
            if not Y_channel_patches:
                raise SystemExit('No training patches found for Y channel; provide training images.')
            Y_all = np.concatenate([t[0] for t in Y_channel_patches], axis=1)
            X_all = np.concatenate([t[1] for t in Y_channel_patches], axis=1)
            D = X_all.shape[0]
            Q = Y_all.shape[0]
            modelY = SparseBayesExpander(D, Q, a_alpha0=20.0, max_iter=200, verbose=True)
            modelY.fit(Y_all, X_all)
            # models[0] will be used for luminance; for I/Q we won't have models (use cubic)
            learned_kernels.append(modelY.M.copy())
            models.append(modelY)
        elif mode == 'yiq-all':
            # convert training images to YIQ and train a model for each Y,I,Q channel
            channel_train_data = [[], [], []]
            for tr in train_imgs:
                H, W = tr.shape[:2]
                Hc = (H // r) * r
                Wc = (W // r) * r
                tr = tr[:Hc, :Wc]
                tri = rgb_to_yiq(tr)
                for ch in range(3):
                    Yt, Xt = extract_patches(tri[:, :, ch], r=r, m=m)
                    channel_train_data[ch].append((Yt, Xt))
            for ch in range(3):
                Ys = [t[0] for t in channel_train_data[ch]]
                Xs = [t[1] for t in channel_train_data[ch]]
                if not Ys:
                    raise SystemExit('No training patches found for channel; provide training images.')
                Y_all = np.concatenate(Ys, axis=1)
                X_all = np.concatenate(Xs, axis=1)
                D = X_all.shape[0]
                Q = Y_all.shape[0]
                model = SparseBayesExpander(D, Q, a_alpha0=20.0, max_iter=200, verbose=True)
                model.fit(Y_all, X_all)
                learned_kernels.append(model.M.copy())
                models.append(model)
    else:
        print('No training images provided.')

    if test_imgs:
        reconstructed_imgs = []
        psnr_list = []
        for tt in test_imgs:
            Ht, Wt = tt.shape[:2]
            Hc = (Ht // r) * r
            Wc = (Wt // r) * r
            tt = tt[:Hc, :Wc]
            rec = np.zeros_like(tt)
            if mode == 'rgb':
                for ch in range(3):
                    low = transform.resize(tt[:, :, ch], (Hc // r, Wc // r), order=3, anti_aliasing=True, preserve_range=True)
                    Hs, Ws = low.shape
                    pad = m // 2
                    low_padded = np.pad(low, pad, mode='edge')
                    out = np.zeros((Hs * r, Ws * r))
                    for i in range(Hs):
                        for j in range(Ws):
                            patch = low_padded[i:i+m, j:j+m].reshape(-1)
                            xr = models[ch].transform_patch(patch)
                            hi = i * r
                            hj = j * r
                            out[hi:hi+r, hj:hj+r] = xr.reshape((r, r))
                    rec[:, :, ch] = out[:tt.shape[0], :tt.shape[1]]
            elif mode == 'yiq-luma':
                # convert test image to YIQ; expand Y with learned model, I/Q with cubic
                t_yiq = rgb_to_yiq(tt)
                # expand Y channel
                lowY = transform.resize(t_yiq[:, :, 0], (Hc // r, Wc // r), order=3, anti_aliasing=True, preserve_range=True)
                Hs, Ws = lowY.shape
                pad = m // 2
                lowY_padded = np.pad(lowY, pad, mode='edge')
                outY = np.zeros((Hs * r, Ws * r))
                for i in range(Hs):
                    for j in range(Ws):
                        patch = lowY_padded[i:i+m, j:j+m].reshape(-1)
                        xr = models[0].transform_patch(patch)
                        hi = i * r
                        hj = j * r
                        outY[hi:hi+r, hj:hj+r] = xr.reshape((r, r))
                # I/Q via cubic: downsample then upsample
                Hlow = tt.shape[0] // r
                Wlow = tt.shape[1] // r
                low_i = transform.resize(t_yiq[:, :, 1], (Hlow, Wlow), order=3, anti_aliasing=True, preserve_range=True)
                low_q = transform.resize(t_yiq[:, :, 2], (Hlow, Wlow), order=3, anti_aliasing=True, preserve_range=True)
                cubic_i = transform.resize(low_i, (tt.shape[0], tt.shape[1]), order=3, anti_aliasing=True, preserve_range=True)
                cubic_q = transform.resize(low_q, (tt.shape[0], tt.shape[1]), order=3, anti_aliasing=True, preserve_range=True)
                # assemble yiq reconstructed
                rec_yiq = np.stack([outY[:tt.shape[0], :tt.shape[1]], cubic_i, cubic_q], axis=2)
                # convert back to RGB for storage/PSNR
                rec_rgb = yiq_to_rgb(rec_yiq)
                # clip and store
                rec = np.clip(rec_rgb, 0, 1)
            elif mode == 'yiq-all':
                # convert test image to YIQ; expand each channel with its learned model
                t_yiq = rgb_to_yiq(tt)
                rec_yiq = np.zeros_like(t_yiq)
                for ch in range(3):
                    low = transform.resize(t_yiq[:, :, ch], (Hc // r, Wc // r), order=3, anti_aliasing=True, preserve_range=True)
                    Hs, Ws = low.shape
                    pad = m // 2
                    low_padded = np.pad(low, pad, mode='edge')
                    out = np.zeros((Hs * r, Ws * r))
                    for i in range(Hs):
                        for j in range(Ws):
                            patch = low_padded[i:i+m, j:j+m].reshape(-1)
                            xr = models[ch].transform_patch(patch)
                            hi = i * r
                            hj = j * r
                            out[hi:hi+r, hj:hj+r] = xr.reshape((r, r))
                    rec_yiq[:, :, ch] = out[:tt.shape[0], :tt.shape[1]]
                # convert back to RGB
                rec_rgb = yiq_to_rgb(rec_yiq)
                rec = np.clip(rec_rgb, 0, 1)
            reconstructed_imgs.append(rec)
            psnr_list.append(sk_psnr(tt, rec, data_range=1.0))
        val_learned = float(np.mean(psnr_list))
        print('Mean PSNR on test set (learned):', val_learned)
        # take first test image for visualization
        img = test_imgs[0]
        reconstructed = reconstructed_imgs[0]
    else:
        parser.error('No test images provided. Please supply --test-dir with images to run the demo (no astronaut fallback).')

    # compute PSNR for learned expander (standard single-channel convention)
    val_learned = sk_psnr(img, reconstructed, data_range=1.0)
    print('PSNR RGB reconstructed vs original (learned):', val_learned)

    # prepare cubic baseline: downsample then upsample with cubic (explicit shapes)
    Hlow = img.shape[0] // r
    Wlow = img.shape[1] // r
    low_rgb = transform.resize(img, (Hlow, Wlow, 3), order=3, anti_aliasing=True, preserve_range=True)
    cubic = transform.resize(low_rgb, img.shape, order=3, anti_aliasing=True, preserve_range=True)
    # ensure in [0,1]
    low_rgb = np.clip(low_rgb, 0, 1)
    cubic = np.clip(cubic, 0, 1)
    val_cubic = sk_psnr(img, cubic, data_range=1.0)
    print('PSNR cubic baseline:', val_cubic)

    # save comparison images
    plt.imsave(out_dir / 'original.png', img)
    plt.imsave(out_dir / 'low_res.png', low_rgb)
    plt.imsave(out_dir / 'cubic.png', np.clip(cubic, 0, 1))
    plt.imsave(out_dir / 'learned.png', np.clip(reconstructed, 0, 1))

    # visualize learned supports per channel (label depending on mode)
    if mode == 'rgb':
        for ch, M_k in enumerate(learned_kernels):
            support = np.mean(np.abs(M_k), axis=0)
            support_map = support.reshape((m, m))
            plt.figure(figsize=(4, 4))
            plt.imshow(support_map, cmap='hot')
            plt.colorbar()
            plt.title(f'Channel {ch} learned support (mean abs weight)')
            plt.savefig(out_dir / f'support_ch{ch}.png')
            plt.close()
    else:
        if mode == 'yiq-luma':
            # only Y was learned (models[0])
            M_k = learned_kernels[0]
            support = np.mean(np.abs(M_k), axis=0)
            support_map = support.reshape((m, m))
            plt.figure(figsize=(4, 4))
            plt.imshow(support_map, cmap='hot')
            plt.colorbar()
            plt.title('Learned support (Y channel)')
            plt.savefig(out_dir / 'support_Y.png')
            plt.close()
        elif mode == 'yiq-all':
            names = ['Y', 'I', 'Q']
            for ch, name in enumerate(names):
                M_k = learned_kernels[ch]
                support = np.mean(np.abs(M_k), axis=0)
                support_map = support.reshape((m, m))
                plt.figure(figsize=(4, 4))
                plt.imshow(support_map, cmap='hot')
                plt.colorbar()
                plt.title(f'Learned support ({name} channel)')
                plt.savefig(out_dir / f'support_{name}.png')
                plt.close()

    # show a small summary figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[1].imshow(low_rgb)
    axes[1].set_title('Low-res')
    axes[2].imshow(np.clip(cubic, 0, 1))
    axes[2].set_title(f'Cubic {val_cubic:.2f} dB')
    axes[3].imshow(np.clip(reconstructed, 0, 1))
    axes[3].set_title(f'Learned ({mode}) {val_learned:.2f} dB')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / 'comparison.png')
    print('Saved outputs to', out_dir)


if __name__ == '__main__':
    main()
