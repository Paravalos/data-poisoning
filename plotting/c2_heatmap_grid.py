"""C2 heatmap grid — mean cooperation (I_A+I_B) and asymmetry (|I_A-I_B|) over motif × stratum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c2_loader import (  # noqa: E402
    MOTIF_ORDER,
    REPO_ROOT,
    SOURCE_STRATUM_ORDER,
    aggregate_by_cell,
    compute_c2_interaction_frame,
)

DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'c2_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'plots' / 'c2_heatmap_grid.png'
DEFAULT_TITLE = 'C2 main effects: motif × source stratum'


def _pivot(summary, value_col):
    matrix = summary.pivot(index='motif_label', columns='source_stratum', values=value_col)
    matrix = matrix.reindex(index=list(MOTIF_ORDER), columns=list(SOURCE_STRATUM_ORDER))
    return matrix


def _draw_heatmap(ax, matrix, *, cmap, vmin, vmax, title, value_fmt='{:+.1f}'):
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if np.isnan(value):
                continue
            bg_norm = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = 'white' if bg_norm < 0.25 or bg_norm > 0.75 else 'black'
            ax.text(j, i, value_fmt.format(value), ha='center', va='center',
                    fontsize=9, color=text_color)
    return image


def create_heatmap_grid_plot(frame, output_path, *, title=DEFAULT_TITLE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = aggregate_by_cell(frame, ['i_sum', 'i_asym'])
    coop_matrix = _pivot(summary, 'i_sum')
    asym_matrix = _pivot(summary, 'i_asym')

    fig, (ax_coop, ax_asym) = plt.subplots(1, 2, figsize=(13, 5.5))

    coop_abs = float(np.nanmax(np.abs(coop_matrix.to_numpy(dtype=float))))
    coop_abs = max(coop_abs, 1e-6)
    coop_image = _draw_heatmap(
        ax_coop, coop_matrix, cmap='RdBu_r', vmin=-coop_abs, vmax=coop_abs,
        title='Cooperation: mean I_A + I_B (pp)',
    )
    fig.colorbar(coop_image, ax=ax_coop, fraction=0.046, pad=0.04).set_label('percentage points')

    asym_max = float(np.nanmax(asym_matrix.to_numpy(dtype=float)))
    asym_max = max(asym_max, 1e-6)
    asym_image = _draw_heatmap(
        ax_asym, asym_matrix, cmap='viridis', vmin=0.0, vmax=asym_max,
        title='Asymmetry: mean |I_A - I_B| (pp)', value_fmt='{:.1f}',
    )
    fig.colorbar(asym_image, ax=ax_asym, fraction=0.046, pad=0.04).set_label('percentage points')

    for ax in (ax_coop, ax_asym):
        ax.set_xlabel('d(A, B) stratum')
        ax.set_ylabel('motif')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main():
    args = _build_argument_parser().parse_args()
    frame = compute_c2_interaction_frame(args.experiment)
    output_path = create_heatmap_grid_plot(frame, args.output)
    summary = aggregate_by_cell(frame, ['i_sum', 'i_asym'])
    print(summary.to_string(index=False))
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
