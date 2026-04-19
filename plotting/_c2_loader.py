"""Shared loading and interaction-score computation for C2 plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_STRATUM_ORDER = ('near', 'mid_near', 'medium', 'mid_far', 'far')
MOTIF_ORDER = (
    'own_aligned_easy',
    'own_aligned_far_apart',
    'cross_aligned_swapped',
    'both_hard_far',
)


def _resolve_repo_path(path_str, repo_root=REPO_ROOT):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def _load_experiment(experiment_path):
    with Path(experiment_path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _solo_mean_adv_confidence(experiment, repo_root):
    solo_means = {}
    for job in experiment.get('solo_jobs', []):
        csv_path = _resolve_repo_path(job['output_path'], repo_root)
        frame = pd.read_csv(csv_path)
        attacker_id = job['attacker']['attacker_id']
        attacker_rows = frame[frame['attacker_id'] == attacker_id]
        if attacker_rows.empty:
            raise KeyError(f'Attacker {attacker_id} not found in {csv_path}.')
        solo_means[attacker_id] = float(attacker_rows['adv_confidence'].mean())
    return solo_means


def _dual_mean_adv_confidence(csv_path):
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby('attacker_id', as_index=False)['adv_confidence'].mean()
    return {row['attacker_id']: float(row['adv_confidence']) for _, row in grouped.iterrows()}


def compute_c2_interaction_frame(experiment_path, repo_root=REPO_ROOT):
    """Return a DataFrame with one row per dual job and all geometry + interaction fields."""
    repo_root = Path(repo_root)
    experiment = _load_experiment(experiment_path)
    solo_means = _solo_mean_adv_confidence(experiment, repo_root)
    class_names = experiment.get('class_names', [])

    rows = []
    for job in experiment.get('dual_jobs', []):
        attackers = job.get('attackers', [])
        if len(attackers) != 2:
            continue
        attacker_a, attacker_b = attackers[0], attackers[1]
        csv_path = _resolve_repo_path(job['output_path'], repo_root)
        dual_means = _dual_mean_adv_confidence(csv_path)

        a_id = attacker_a['attacker_id']
        b_id = attacker_b['attacker_id']
        if a_id not in dual_means or b_id not in dual_means:
            raise KeyError(f'Missing attacker in {csv_path}.')
        if a_id not in solo_means or b_id not in solo_means:
            raise KeyError(f'Missing solo output for {a_id} or {b_id}.')

        i_a = 100.0 * (dual_means[a_id] - solo_means[a_id])
        i_b = 100.0 * (dual_means[b_id] - solo_means[b_id])

        def _class_name(idx):
            if class_names and 0 <= int(idx) < len(class_names):
                return class_names[int(idx)]
            return str(idx)

        rows.append(dict(
            pairing_id=job['pairing_id'],
            source_pair_label=job['source_pair_label'],
            source_stratum=job['source_stratum'],
            motif_label=job['motif_label'],
            alignment_type=job.get('alignment_type', ''),
            attacker_a_id=a_id,
            attacker_b_id=b_id,
            attacker_a_class_name=_class_name(attacker_a['source_class']),
            attacker_b_class_name=_class_name(attacker_b['source_class']),
            target_a_class_name=job.get('target_a_class_name', _class_name(job.get('target_a_class', -1))),
            target_b_class_name=job.get('target_b_class_name', _class_name(job.get('target_b_class', -1))),
            source_source_distance=float(job['source_source_distance']),
            target_target_distance=float(job['target_target_distance']),
            a_self=float(job['a_self']),
            b_self=float(job['b_self']),
            a_cross=float(job['a_cross']),
            b_cross=float(job['b_cross']),
            cross_alignment_gap=float(job['cross_alignment_gap']),
            i_a=i_a,
            i_b=i_b,
            i_sum=i_a + i_b,
            i_asym=abs(i_a - i_b),
        ))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No dual jobs found in experiment.')
    frame['source_stratum'] = pd.Categorical(
        frame['source_stratum'], categories=list(SOURCE_STRATUM_ORDER), ordered=True,
    )
    frame['motif_label'] = pd.Categorical(
        frame['motif_label'], categories=list(MOTIF_ORDER), ordered=True,
    )
    return frame.sort_values(['motif_label', 'source_stratum', 'pairing_id']).reset_index(drop=True)


def aggregate_by_cell(frame, value_columns):
    """Mean of each value column, grouped by (motif_label, source_stratum). Returns long-format frame."""
    grouped = frame.groupby(['motif_label', 'source_stratum'], observed=False)
    summary = grouped[list(value_columns)].mean().reset_index()
    return summary
