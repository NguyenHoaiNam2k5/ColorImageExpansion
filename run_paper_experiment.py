import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from demo import evaluate_models, fit_models, load_images_from_dir


DEFAULT_A_ALPHA0 = '5,10,20,40,70,100,130,170,210'
DEFAULT_MODES = 'rgb,yiq-luma,yiq-all'
MODE_LABELS = {
    'rgb': 'RGB',
    'yiq-luma': 'Y + cubic',
    'yiq-all': 'YIQ',
}


def parse_csv_floats(text):
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def parse_csv_modes(text):
    valid = {'rgb', 'yiq-luma', 'yiq-all'}
    modes = [item.strip() for item in text.split(',') if item.strip()]
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f'unknown modes: {bad}; valid modes are {sorted(valid)}')
    return modes


def parse_args():
    parser = argparse.ArgumentParser(description='Run the paper-style sparse Bayesian color expansion sweep.')
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--test-dir', type=str, default='data/test')
    parser.add_argument('--output-dir', type=str, default='paper_outputs')
    parser.add_argument('--modes', type=str, default=DEFAULT_MODES)
    parser.add_argument('--a-alpha0-values', type=str, default=DEFAULT_A_ALPHA0)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--m', type=int, default=11)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--symmetry', choices=['none', 'h', 'v', 'hv'], default='hv')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def make_model_args(args, a_alpha0):
    return SimpleNamespace(
        r=args.r,
        m=args.m,
        a_alpha0=a_alpha0,
        b_alpha0=2e-8,
        a_beta0=1e-6,
        b_beta0=1e-6,
        max_iter=args.max_iter,
        tol=args.tol,
        symmetry=args.symmetry,
        verbose=args.verbose,
    )


def save_paper_style_plot(rows, modes, out_dir):
    plt.figure(figsize=(6, 4))
    for mode in modes:
        mode_rows = [row for row in rows if row['mode'] == mode]
        mode_rows.sort(key=lambda row: row['mean_support_size'])
        if not mode_rows:
            continue
        plt.plot(
            [row['mean_support_size'] for row in mode_rows],
            [row['mean_psnr_learned'] for row in mode_rows],
            marker='o',
            label=MODE_LABELS.get(mode, mode),
        )

    plt.xlabel('Support size (pixel)')
    plt.ylabel('PSNR (dB)')
    plt.xticks([9, 25, 49, 81, 121])
    plt.xlim(9, 121)
    plt.title('Performance of image expansion filters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'psnr_vs_support.png', dpi=150)
    plt.close()


def paper_support_size(model):
        return float(model.active_mask().sum(axis=1).mean())



def save_support_grid(models, mode, a_alpha0, m, out_dir, d = 0):
    if mode == 'rgb':
        names = ['R', 'G', 'B']
        filename = f'fig4_rgb_support_a{a_alpha0:g}.png'
        title = f'RGB learned support, a_alpha0={a_alpha0:g}'
    elif mode == 'yiq-all':
        names = ['Y', 'I', 'Q']
        filename = f'fig5_yiq_support_a{a_alpha0:g}.png'
        title = f'YIQ learned support, a_alpha0={a_alpha0:g}'
    else:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for ax, name, model in zip(axes, names, models):
        # support_map = model.support_union_mask().reshape((m, m)).astype(float)
        support_map = model.active_mask()[d].reshape((m, m)).astype(float)
        ax.imshow(support_map, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    modes = parse_csv_modes(args.modes)
    a_values = parse_csv_floats(args.a_alpha0_values)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = load_images_from_dir(args.train_dir)
    test_imgs = load_images_from_dir(args.test_dir)
    if not train_imgs:
        raise SystemExit(f'no training images found in {args.train_dir}')
    if not test_imgs:
        raise SystemExit(f'no test images found in {args.test_dir}')

    rows = []
    for mode in modes:
        for a_alpha0 in a_values:
            print(f'Running mode={mode}, a_alpha0={a_alpha0}')
            model_args = make_model_args(args, a_alpha0)
            models = fit_models(train_imgs, mode, model_args)

            for ch, model in enumerate(models):
                print(
                    f'mode={mode}, a_alpha0={a_alpha0}, channel={ch}, '
                    f'row_support_sizes={model.active_mask().sum(axis=1)}, '
                    f'union_support_size={model.support_size()}'
                )

            if a_alpha0 in (20.0, 130.0):
                save_support_grid(models, mode, a_alpha0, args.m, out_dir)

            learned_scores, cubic_scores, _ = evaluate_models(models, test_imgs, mode, args.r, args.m)
            # support_sizes = [model.support_size() for model in models]

            row_support_sizes = [paper_support_size(model) for model in models]
            union_support_sizes = [model.support_size() for model in models]

            row = {
                'mode': mode,
                'a_alpha0': a_alpha0,
                'mean_psnr_learned': float(np.mean(learned_scores)),
                'mean_psnr_cubic': float(np.mean(cubic_scores)),
                'mean_support_size': float(np.mean(row_support_sizes)),
                'support_sizes': ';'.join(f'{s:g}' for s in row_support_sizes),
                'union_support_sizes': ';'.join(str(s) for s in union_support_sizes),
                'n_train_images': len(train_imgs),
                'n_test_images': len(test_imgs),
                'r': args.r,
                'm': args.m,
            }
            rows.append(row)
            print(row)

    csv_path = out_dir / 'results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_paper_style_plot(rows, modes, out_dir)

    print(f'Saved {csv_path}')
    print(f'Saved {out_dir / "psnr_vs_support.png"}')


if __name__ == '__main__':
    main()
