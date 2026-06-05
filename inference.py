import numpy as np

from demo import cubic_expand, rgb_to_yiq, yiq_to_rgb


def _as_float_rgb_image(img):
    """Return an RGB float64 image from a grayscale/RGB low-resolution input."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError('low_img must be a 2-D grayscale image or an H x W x 3 RGB image')

    if arr.size and np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _resolve_out_shape(low_img, r, out_shape):
    base_shape = (low_img.shape[0] * r, low_img.shape[1] * r)
    if out_shape is None:
        return base_shape

    if len(out_shape) == 3:
        out_shape = out_shape[:2]
    if len(out_shape) != 2:
        raise ValueError('out_shape must be (height, width) or (height, width, channels)')

    out_shape = tuple(int(v) for v in out_shape)
    if out_shape[0] <= 0 or out_shape[1] <= 0:
        raise ValueError('out_shape dimensions must be positive')
    if out_shape[0] > base_shape[0] or out_shape[1] > base_shape[1]:
        raise ValueError(
            f'out_shape={out_shape} is larger than the learned expansion size {base_shape}; '
            'use a smaller/equal crop shape or a larger expansion factor'
        )
    return out_shape


def _validate_channel_model(model, r, m):
    expected_d = r * r
    expected_q = m * m
    if getattr(model, 'D', expected_d) != expected_d:
        raise ValueError(f'model.D must be {expected_d} for r={r}')
    if getattr(model, 'Q', expected_q) != expected_q:
        raise ValueError(f'model.Q must be {expected_q} for m={m}')


def _validate_model_count(models, expected, mode):
    models = list(models)
    if len(models) != expected:
        raise ValueError(f'mode={mode!r} requires {expected} model(s), got {len(models)}')
    return models


def expand_channel_from_low(model, low_channel, r=2, m=11, out_shape=None):
    """Expand one already-low-resolution channel with a learned filter.

    Unlike ``expand_channel`` in ``demo.py``, this function does not synthesize a
    low-resolution image from a high-resolution reference image. It directly uses
    ``low_channel`` as the observed low-resolution input and applies the learned
    posterior-mean filter patch by patch.

    Parameters
    ----------
    model : SparseBayesExpander
        Trained one-channel model.
    low_channel : ndarray, shape (H, W)
        Low-resolution input channel.
    r : int, default=2
        Expansion factor. Must match the trained model's output patch size.
    m : int, default=11
        Low-resolution patch size. Must match the trained model's input patch size.
    out_shape : tuple, optional
        Optional output crop shape ``(height, width)``. By default the output shape
        is ``(H * r, W * r)``.
    """
    _validate_channel_model(model, r, m)
    low = np.asarray(low_channel, dtype=np.float64)
    if low.ndim != 2:
        raise ValueError('low_channel must be a 2-D array')

    target_shape = _resolve_out_shape(low, r, out_shape)
    pad = m // 2
    low_padded = np.pad(low, pad, mode='edge')
    h_low, w_low = low.shape
    out = np.zeros((h_low * r, w_low * r), dtype=np.float64)

    for i in range(h_low):
        for j in range(w_low):
            patch = low_padded[i:i + m, j:j + m].reshape(-1)
            out[i * r:(i + 1) * r, j * r:(j + 1) * r] = (
                model.transform_patch(patch).reshape((r, r))
            )

    return out[:target_shape[0], :target_shape[1]]


def expand_lr_image(models, low_img, mode, r=2, m=11, out_shape=None, clip=True):
    """Expand an already-low-resolution RGB image with trained models.

    Parameters
    ----------
    models : sequence of SparseBayesExpander
        Trained channel models returned by ``fit_models``.
    low_img : ndarray, shape (H, W), (H, W, 1), or (H, W, 3)
        Observed low-resolution image. Integer images in [0, 255] are converted
        to float images in [0, 1].
    mode : {'rgb', 'yiq-luma', 'yiq-all'}
        Same color scheme used when training the models.
    r : int, default=2
        Expansion factor.
    m : int, default=11
        Low-resolution patch size.
    out_shape : tuple, optional
        Optional output crop shape ``(height, width)`` or ``(height, width, 3)``.
        By default the output shape is ``(H * r, W * r, 3)``.
    clip : bool, default=True
        If true, clip the final RGB output to [0, 1].
    """
    low_rgb = _as_float_rgb_image(low_img)
    target_shape = _resolve_out_shape(low_rgb, r, out_shape)

    if mode == 'rgb':
        models = _validate_model_count(models, 3, mode)
        rec = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float64)
        for ch, model in enumerate(models):
            rec[:, :, ch] = expand_channel_from_low(
                model, low_rgb[:, :, ch], r=r, m=m, out_shape=target_shape
            )
    elif mode == 'yiq-luma':
        models = _validate_model_count(models, 1, mode)
        low_yiq = rgb_to_yiq(low_rgb)
        out_y = expand_channel_from_low(
            models[0], low_yiq[:, :, 0], r=r, m=m, out_shape=target_shape
        )
        out_i = cubic_expand(low_yiq[:, :, 1], target_shape)
        out_q = cubic_expand(low_yiq[:, :, 2], target_shape)
        rec = yiq_to_rgb(np.stack([out_y, out_i, out_q], axis=2))
    elif mode == 'yiq-all':
        models = _validate_model_count(models, 3, mode)
        low_yiq = rgb_to_yiq(low_rgb)
        rec_yiq = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float64)
        for ch, model in enumerate(models):
            rec_yiq[:, :, ch] = expand_channel_from_low(
                model, low_yiq[:, :, ch], r=r, m=m, out_shape=target_shape
            )
        rec = yiq_to_rgb(rec_yiq)
    else:
        raise ValueError(f'unknown mode: {mode}')

    if clip:
        rec = np.clip(rec, 0.0, 1.0)
    return rec


def cubic_baseline_from_low(low_img, r=2, out_shape=None, clip=True):
    """Cubic baseline for an already-low-resolution RGB/grayscale image."""
    low_rgb = _as_float_rgb_image(low_img)
    target_shape = _resolve_out_shape(low_rgb, r, out_shape)
    rec = cubic_expand(low_rgb, target_shape)
    if clip:
        rec = np.clip(rec, 0.0, 1.0)
    return rec
