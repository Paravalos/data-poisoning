"""Create the C1 grouped bar chart for solo vs together confidences."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c1' / 'c1_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c1' / 'plots' / 'c1_preliminary_grouped_bar_adv_confidence.png'
DEFAULT_TITLE = 'C1 preliminary: solo vs together confidence when both attackers target airplane, attacker A = dog'


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


def compute_grouped_bar_summary(experiment_path=DEFAULT_EXPERIMENT_PATH, include_self_pair=False):
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

        fixed_dual = dual_lookup[fixed_attacker_id]
        partner_dual = dual_lookup[partner_attacker_id]
        fixed_solo = solo_lookup[fixed_attacker_id]
        partner_solo = solo_lookup[partner_attacker_id]

        partner_name = class_names[partner_class_idx]
        partner_rows[partner_name].append(dict(
            partner_class=int(partner_class_idx),
            partner_class_name=partner_name,
            a_solo=float(fixed_solo['adv_confidence']),
            b_solo=float(partner_solo['adv_confidence']),
            a_together=float(fixed_dual['adv_confidence']),
            b_together=float(partner_dual['adv_confidence']),
            dog_partner_distance=float(job['source_source_distance']),
            partner_target_distance=float(partner_dual['source_target_distance']),
        ))

    if not partner_rows:
        raise ValueError('No dual partner rows were found for the requested configuration.')

    summary_rows = []
    for partner_name, entries in partner_rows.items():
        summary_rows.append(dict(
            partner_class=int(entries[0]['partner_class']),
            partner_class_name=partner_name,
            dog_partner_distance=float(np.mean([entry['dog_partner_distance'] for entry in entries])),
            partner_target_distance=float(np.mean([entry['partner_target_distance'] for entry in entries])),
            a_solo_percentage_points=100.0 * float(np.mean([entry['a_solo'] for entry in entries])),
            b_solo_percentage_points=100.0 * float(np.mean([entry['b_solo'] for entry in entries])),
            a_together_percentage_points=100.0 * float(np.mean([entry['a_together'] for entry in entries])),
            b_together_percentage_points=100.0 * float(np.mean([entry['b_together'] for entry in entries])),
            repeats=len(entries),
        ))

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by=['dog_partner_distance', 'partner_class_name'],
        kind='mergesort',
    ).reset_index(drop=True)
    return summary_frame, fixed_attacker_name, target_class_name


def create_grouped_bar_plot(
    summary_frame,
    output_path,
    *,
    fixed_attacker_name='dog',
    target_class_name='airplane',
    title=DEFAULT_TITLE,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_positions = np.arange(len(summary_frame))
    width = 0.2

    bar_specs = [
        ('a_solo_percentage_points', f'{fixed_attacker_name} solo', '#4c78a8'),
        ('b_solo_percentage_points', 'partner solo', '#72b7b2'),
        ('a_together_percentage_points', f'{fixed_attacker_name} together', '#f58518'),
        ('b_together_percentage_points', 'partner together', '#e45756'),
    ]
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for (column, label, color), offset in zip(bar_specs, offsets):
        ax.bar(x_positions + offset * width, summary_frame[column], width=width, label=label, color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_frame['partner_class_name'], rotation=20, ha='right')
    ax.set_ylabel(f'Mean P({target_class_name}) on target image (percentage points)')
    ax.set_xlabel('Partner class')
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(description='Create the C1 grouped bar chart for mean adv_confidence.')
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH), help='Path to the C1 experiment JSON.')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='Where to save the plot.')
    parser.add_argument('--include-self-pair', action='store_true', help='Include the dog-vs-dog self-pair.')
    return parser


def main():
    args = _build_argument_parser().parse_args()
    summary_frame, fixed_attacker_name, target_class_name = compute_grouped_bar_summary(
        experiment_path=args.experiment,
        include_self_pair=args.include_self_pair,
    )
    output_path = create_grouped_bar_plot(
        summary_frame,
        args.output,
        fixed_attacker_name=fixed_attacker_name,
        target_class_name=target_class_name,
    )
    print(summary_frame.to_string(index=False))
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
