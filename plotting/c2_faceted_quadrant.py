"""C2 faceted I_A vs I_B quadrant scatter — one panel per motif, color by source stratum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c2_loader import (  # noqa: E402
    MOTIF_ORDER,
    REPO_ROOT,
    SOURCE_STRATUM_ORDER,
    aggregate_by_cell,
    compute_c2_interaction_frame,
)

DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'c2_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'plots' / 'c2_faceted_quadrant.png'
DEFAULT_TITLE = 'C2: I_A vs I_B by motif (color = d(A, B) stratum)'

STRATUM_COLORS = {
    'near': '#2166ac',
    'mid_near': '#67a9cf',
    'medium': '#737373',
    'mid_far': '#ef8a62',
    'far': '#b2182b',
}


def _scale_sizes(values, min_size=80.0, max_size=320.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.max(), values.min()):
        return np.full(values.shape, (min_size + max_size) / 2.0)
    normalized = (values - values.min()) / (values.max() - values.min())
    return min_size + normalized * (max_size - min_size)


def create_faceted_quadrant_plot(frame, output_path, *, title=DEFAULT_TITLE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = aggregate_by_cell(frame, ['i_a', 'i_b', 'target_target_distance']).dropna(subset=['i_a', 'i_b'])

    all_values = np.concatenate([
        frame['i_a'].to_numpy(dtype=float),
        frame['i_b'].to_numpy(dtype=float),
    ])
    limit = float(np.max(np.abs(all_values))) if all_values.size else 1.0
    limit = max(limit * 1.2, 1.0)

    sizes = _scale_sizes(frame['target_target_distance'].to_numpy())
    frame = frame.assign(_marker_size=sizes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for ax, motif in zip(flat_axes, MOTIF_ORDER):
        panel = frame[frame['motif_label'] == motif]
        for stratum in SOURCE_STRATUM_ORDER:
            subset = panel[panel['source_stratum'] == stratum]
            if subset.empty:
                continue
            ax.scatter(
                subset['i_a'],
                subset['i_b'],
                c=[to_rgba(STRATUM_COLORS[stratum])],
                s=subset['_marker_size'],
                edgecolor='black',
                linewidth=0.6,
                alpha=0.85,
                label=stratum,
            )

        ax.axhline(0.0, color='black', linewidth=0.9)
        ax.axvline(0.0, color='black', linewidth=0.9)
        ax.plot([-limit, limit], [-limit, limit], color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.5)
        ax.set_title(motif, fontsize=11)

        label_kwargs = dict(fontsize=8, color='dimgray',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.2))
        ax.text(0.96 * limit, 0.96 * limit, 'collaboration', ha='right', va='top', **label_kwargs)
        ax.text(-0.96 * limit, -0.96 * limit, 'mutual degradation', ha='left', va='bottom', **label_kwargs)
        ax.text(-0.96 * limit, 0.96 * limit, 'A down, B up', ha='left', va='top', **label_kwargs)
        ax.text(0.96 * limit, -0.96 * limit, 'A up, B down', ha='right', va='bottom', **label_kwargs)

    for ax in axes[-1, :]:
        ax.set_xlabel('I_A (attacker A, percentage points)')
    for ax in axes[:, 0]:
        ax.set_ylabel('I_B (attacker B, percentage points)')

    stratum_handles = [
        plt.scatter([], [], c=STRATUM_COLORS[stratum], s=120, edgecolor='black', linewidth=0.6)
        for stratum in SOURCE_STRATUM_ORDER
    ]
    size_handles = [
        plt.scatter([], [], s=80, color='white', edgecolor='black'),
        plt.scatter([], [], s=320, color='white', edgecolor='black'),
    ]
    legend1 = fig.legend(
        stratum_handles, list(SOURCE_STRATUM_ORDER),
        title='d(A, B) stratum', loc='upper right', bbox_to_anchor=(0.995, 0.97), frameon=False,
    )
    fig.add_artist(legend1)
    fig.legend(
        size_handles, ['close targets', 'far targets'],
        title='d(T_A, T_B)', loc='upper right', bbox_to_anchor=(0.995, 0.80), frameon=False,
    )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
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
    output_path = create_faceted_quadrant_plot(frame, args.output)
    print(frame[['motif_label', 'source_stratum', 'i_a', 'i_b', 'i_sum']].to_string(index=False))
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
