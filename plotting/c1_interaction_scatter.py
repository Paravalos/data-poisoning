"""Create the C1 interaction scatter plot for the fixed-dog sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c1' / 'c1_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c1' / 'plots' / 'c1_preliminary_interaction_scatter.png'
DEFAULT_TITLE = 'C1 preliminary: interaction when both attackers target airplane, attacker A = dog'


def _resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_experiment(experiment_path):
    with Path(experiment_path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _read_confidence_rows(csv_path):
    frame = pd.read_csv(csv_path)
    required_columns = {
        'attacker_id',
        'source_class',
        'source_class_name',
        'target_index',
        'source_target_distance',
        'adv_confidence',
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns in {csv_path}: {sorted(missing)}')
    return frame


def _mean_attacker_rows(csv_path):
    frame = _read_confidence_rows(csv_path)
    aggregations = dict(
        source_class=('source_class', 'first'),
        source_class_name=('source_class_name', 'first'),
        target_index=('target_index', 'first'),
        source_target_distance=('source_target_distance', 'first'),
        adv_confidence=('adv_confidence', 'mean'),
    )
    if 'source_source_distance' in frame.columns:
        aggregations['source_source_distance'] = ('source_source_distance', 'first')
    grouped = frame.groupby('attacker_id', as_index=False).agg(**aggregations)
    return {
        row['attacker_id']: {
            key: row[key]
            for key in grouped.columns
            if key != 'attacker_id'
        }
        for _, row in grouped.iterrows()
    }


def _build_solo_lookup(experiment):
    solo_lookup = {}
    for job in experiment.get('solo_jobs', []):
        csv_path = _resolve_repo_path(job['output_path'])
        attacker_rows = _mean_attacker_rows(csv_path)
        attacker_id = job['attacker']['attacker_id']
        if attacker_id not in attacker_rows:
            raise KeyError(f'Attacker {attacker_id} not found in {csv_path}.')
        solo_lookup[attacker_id] = attacker_rows[attacker_id]
    return solo_lookup


def compute_interaction_summary(experiment_path=DEFAULT_EXPERIMENT_PATH, include_self_pair=False):
    experiment = _load_experiment(experiment_path)
    class_names = experiment['class_names']
    metadata = experiment['metadata']
    fixed_attacker_idx = int(metadata['fixed_attacker_a_source_class'])
    target_class_idx = int(metadata['shared_target_class'])
    fixed_attacker_name = class_names[fixed_attacker_idx]
    target_class_name = class_names[target_class_idx]

    solo_lookup = _build_solo_lookup(experiment)
    partner_rows = defaultdict(list)

    for job in experiment.get('dual_jobs', []):
        attackers = job.get('attackers', [])
        if len(attackers) != 2:
            continue

        fixed_attacker = attackers[0]
        partner_attacker = attackers[1]
        partner_class_idx = int(partner_attacker['source_class'])
        if not include_self_pair and partner_class_idx == fixed_attacker_idx:
            continue

        dual_csv_path = _resolve_repo_path(job['output_path'])
        dual_lookup = _mean_attacker_rows(dual_csv_path)
        fixed_attacker_id = fixed_attacker['attacker_id']
        partner_attacker_id = partner_attacker['attacker_id']

        if fixed_attacker_id not in dual_lookup:
            raise KeyError(f'Attacker {fixed_attacker_id} not found in {dual_csv_path}.')
        if partner_attacker_id not in dual_lookup:
            raise KeyError(f'Attacker {partner_attacker_id} not found in {dual_csv_path}.')
        if fixed_attacker_id not in solo_lookup:
            raise KeyError(f'Attacker {fixed_attacker_id} not found in solo outputs.')
        if partner_attacker_id not in solo_lookup:
            raise KeyError(f'Attacker {partner_attacker_id} not found in solo outputs.')

        fixed_dual = dual_lookup[fixed_attacker_id]
        partner_dual = dual_lookup[partner_attacker_id]
        fixed_solo = solo_lookup[fixed_attacker_id]
        partner_solo = solo_lookup[partner_attacker_id]

        partner_name = class_names[partner_class_idx]
        partner_rows[partner_name].append(dict(
            partner_class=int(partner_class_idx),
            partner_class_name=partner_name,
            fixed_target_index=int(fixed_dual['target_index']),
            partner_target_index=int(partner_dual['target_index']),
            i_a=float(fixed_dual['adv_confidence'] - fixed_solo['adv_confidence']),
            i_b=float(partner_dual['adv_confidence'] - partner_solo['adv_confidence']),
            dog_partner_distance=float(job['source_source_distance']),
            partner_target_distance=float(partner_dual['source_target_distance']),
            pairing_id=job['pairing_id'],
        ))

    per_image_frame = pd.DataFrame(
        entry
        for entries in partner_rows.values()
        for entry in entries
    )
    if per_image_frame.empty:
        raise ValueError('No dual partner rows were found for the requested configuration.')

    summary_rows = []
    for partner_name, entries in partner_rows.items():
        summary_rows.append(dict(
            partner_class=int(entries[0]['partner_class']),
            partner_class_name=partner_name,
            i_a_percentage_points=100.0 * float(np.mean([entry['i_a'] for entry in entries])),
            i_b_percentage_points=100.0 * float(np.mean([entry['i_b'] for entry in entries])),
            dog_partner_distance=float(np.mean([entry['dog_partner_distance'] for entry in entries])),
            partner_target_distance=float(np.mean([entry['partner_target_distance'] for entry in entries])),
            repeats=len(entries),
        ))

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by=['dog_partner_distance', 'partner_class_name'],
        kind='mergesort',
    ).reset_index(drop=True)
    return summary_frame, per_image_frame, fixed_attacker_name, target_class_name


def _scale_point_sizes(values, min_size=180.0, max_size=700.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if np.allclose(values.max(), values.min()):
        return np.full(values.shape, (min_size + max_size) / 2.0)
    normalized = (values - values.min()) / (values.max() - values.min())
    return min_size + normalized * (max_size - min_size)


def create_interaction_scatter_plot(
    summary_frame,
    output_path,
    *,
    fixed_attacker_name='dog',
    target_class_name='airplane',
    title=DEFAULT_TITLE,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = _scale_point_sizes(summary_frame['partner_target_distance'].to_numpy())
    color_values = summary_frame['dog_partner_distance'].to_numpy(dtype=float)
    norm = Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        summary_frame['i_a_percentage_points'],
        summary_frame['i_b_percentage_points'],
        c=color_values,
        s=sizes,
        cmap='coolwarm',
        norm=norm,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.9,
    )

    axis_values = np.concatenate([
        summary_frame['i_a_percentage_points'].to_numpy(dtype=float),
        summary_frame['i_b_percentage_points'].to_numpy(dtype=float),
    ])
    limit = float(np.max(np.abs(axis_values))) if axis_values.size else 1.0
    limit = max(limit * 1.35, 1.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.axhline(0.0, color='black', linewidth=1.0)
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.plot([-limit, limit], [-limit, limit], color='black', linewidth=1.0, linestyle='-')

    label_offsets = [
        (8, 8),
        (8, -12),
        (-10, 8),
        (-10, -12),
    ]
    for row_idx, (_, row) in enumerate(summary_frame.iterrows()):
        dx, dy = label_offsets[row_idx % len(label_offsets)]
        ax.annotate(
            row['partner_class_name'],
            (row['i_a_percentage_points'], row['i_b_percentage_points']),
            textcoords='offset points',
            xytext=(dx, dy),
            ha='left' if dx >= 0 else 'right',
            va='bottom' if dy >= 0 else 'top',
        )

    label_kwargs = dict(
        fontsize=10,
        color='dimgray',
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
    )
    ax.text(0.98 * limit, 0.98 * limit, 'collaboration', ha='right', va='top', **label_kwargs)
    ax.text(-0.98 * limit, -0.98 * limit, 'mutual degradation', ha='left', va='bottom', **label_kwargs)
    ax.text(-0.98 * limit, 0.98 * limit, 'A degrades, B improves', ha='left', va='top', **label_kwargs)
    ax.text(0.98 * limit, -0.98 * limit, 'A improves, B degrades', ha='right', va='bottom', **label_kwargs)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(f'd({fixed_attacker_name}, partner)')

    size_handles = [
        ax.scatter([], [], s=180.0, color='white', edgecolor='black'),
        ax.scatter([], [], s=700.0, color='white', edgecolor='black'),
    ]
    ax.legend(
        size_handles,
        [
            f'close to {target_class_name}',
            f'far from {target_class_name}',
        ],
        title=f'd(partner, {target_class_name})',
        loc='upper left',
        bbox_to_anchor=(0.02, 0.93),
        borderaxespad=0.0,
        frameon=False,
    )

    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_xlabel(f'I_A ({fixed_attacker_name} interaction score, percentage points)')
    ax.set_ylabel('I_B (partner interaction score, percentage points)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(description='Create the C1 interaction scatter plot for the fixed-dog sweep.')
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH), help='Path to the C1 experiment JSON.')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='Where to save the plot.')
    parser.add_argument('--include-self-pair', action='store_true', help='Include the dog-vs-dog self-pair.')
    return parser


def main():
    args = _build_argument_parser().parse_args()
    summary_frame, _, fixed_attacker_name, target_class_name = compute_interaction_summary(
        experiment_path=args.experiment,
        include_self_pair=args.include_self_pair,
    )
    output_path = create_interaction_scatter_plot(
        summary_frame,
        args.output,
        fixed_attacker_name=fixed_attacker_name,
        target_class_name=target_class_name,
    )
    print(summary_frame.to_string(index=False))
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
