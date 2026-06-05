# Sparse Bayesian Image Expansion Filters (Kanemura et al. 2009)

This project implements the variational sparse Bayesian learning approach for
color image expansion filters from Kanemura, Maeda, and Ishii, **Learning Color
Image Expansion Filters** (ICIP 2009).

The implementation is organized around the paper's model

```text
x = W y + eps
```

where `y` is an `m x m` low-resolution patch and `x` is the corresponding
`r x r` high-resolution patch. Color expansion is handled by training one
scalar-channel model per RGB/YIQ channel, matching the paper's three schemes:

- `rgb`: train R, G, and B filters independently.
- `yiq-luma`: train Y only; expand I/Q by cubic interpolation.
- `yiq-all`: train Y, I, and Q filters independently.

## What is implemented

- Variational updates for `q(A)`, `q(W)`, and `q(beta)`.
- ARD pruning with the paper's `exp(20)` threshold.
- Paper-style horizontal/vertical symmetry constraints over `(d, q)` coefficient
  pairs: output sub-pixel index `d` and input-patch index `q` are mirrored
  together.
- Low-resolution image generation as an explicit cubic-kernel anti-aliasing blur
  followed by subsampling.
- Training patch extraction that discards boundary low-resolution patches.
- Test-time patch extraction with pixel-replication padding.
- ARD support maps based on active alpha values, not mean absolute weights.
- A paper-style sweep script for PSNR vs. support size.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Dataset layout

Place the training and test images in separate folders:

```text
data/
  train/
    ... images ...
  test/
    ... images ...
```

For the paper's experiment, use USC-SIPI images `#4.1.[01-08]` for training and
`#4.2.[01-07]` for testing. The code accepts `.png`, `.jpg`, `.jpeg`, `.tif`,
`.tiff`, and `.bmp` files.

## Run one demo

```powershell
python demo.py --train-dir data/train --test-dir data/test --mode yiq-all --r 2 --m 11 --a-alpha0 20
```

Outputs are written to `demo_outputs/`:

- reconstructed image from the learned filter
- cubic baseline image
- low-resolution image
- ARD support maps
- comparison figure

## Run the paper-style sweep

```powershell
python run_paper_experiment.py --train-dir data/train --test-dir data/test --r 2 --m 11
```

This runs `rgb`, `yiq-luma`, and `yiq-all` over the default sparsity sweep:

```text
5,10,20,40,70,100,130,170,210
```

Outputs are written to `paper_outputs/`:

- `results.csv`
- `psnr_vs_support.png`, a combined RGB / YIQ / Y + cubic PSNR-vs-support plot
  in the style of Fig. 3 in the paper

## Notes

The paper states that the low-resolution images are made by cubic-kernel
anti-aliasing and subsampling. This repo keeps that preprocessing as two
explicit steps: a separable scaled cubic convolution blur followed by decimation
by the expansion factor. The paper does not specify every numerical detail of
the cubic preprocessing kernel, so small differences from the original ICIP 2009
implementation are still possible.
