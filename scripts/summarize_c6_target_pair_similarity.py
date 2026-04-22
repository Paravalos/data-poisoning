"""Summarize C6 closest-vs-random target-pair similarity results."""

from __future__ import annotations

import argparse
import csv
import os
from statistics import mean

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.experiment import load_experiment


SUMMARY_FIELDS = [
    'experiment_id',
    'pairing_id',
    'condition',
    'pair_bin',
    'selection_method',
    'repeat_slot',
    'pair_number',
    'pair_key',
    'target_pair_left_index',
    'target_pair_right_index',
    'target_class_name',
    'poison_class_name',
    'feature_cosine_distance',
    'feature_cosine_similarity',
    'gradient_cosine',
    'cluster_id',
    'cluster_size',
    'attacker_a_id',
    'attacker_b_id',
    'asr_a_alone',
    'asr_b_alone',
    'asr_a_combined',
    'asr_b_combined',
    'lift_a',
    'lift_b',
    'lift',
    'quadrant',
]


def _resolve_path(path, repo_root):
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _read_csv_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _mean_success(rows, attacker_id, csv_path):
    attacker_rows = [row for row in rows if row.get('attacker_id') == attacker_id]
    if not attacker_rows:
        raise KeyError(f'Attacker {attacker_id} not found in {csv_path}.')
    return mean(float(row['success']) for row in attacker_rows)


def _solo_asr_by_attacker(experiment, repo_root):
    asr_by_attacker = {}
    for job in experiment.get('solo_jobs', []):
        attacker_id = job['attacker']['attacker_id']
        csv_path = _resolve_path(job['output_path'], repo_root)
        asr_by_attacker[attacker_id] = _mean_success(_read_csv_rows(csv_path), attacker_id, csv_path)
    return asr_by_attacker


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _quadrant(lift_a, lift_b):
    sign_a = _sign(lift_a)
    sign_b = _sign(lift_b)
    if sign_a > 0 and sign_b > 0:
        return 'collaboration'
    if sign_a < 0 and sign_b < 0:
        return 'mutual_degradation'
    if sign_a == 0 and sign_b == 0:
        return 'independence'
    if sign_a * sign_b < 0:
        return 'opposition'
    return 'mixed_with_independence'


def summarize_c6_experiment(experiment, repo_root):
    solo_asr = _solo_asr_by_attacker(experiment, repo_root)
    summaries = []

    for job in experiment.get('dual_jobs', []):
        left, right = job['attackers']
        left_id = left['attacker_id']
        right_id = right['attacker_id']
        dual_csv_path = _resolve_path(job['output_path'], repo_root)
        dual_rows = _read_csv_rows(dual_csv_path)

        asr_a_alone = solo_asr[left_id]
        asr_b_alone = solo_asr[right_id]
        asr_a_combined = _mean_success(dual_rows, left_id, dual_csv_path)
        asr_b_combined = _mean_success(dual_rows, right_id, dual_csv_path)
        lift_a = asr_a_combined - asr_a_alone
        lift_b = asr_b_combined - asr_b_alone

        summaries.append(dict(
            experiment_id=experiment['experiment_id'],
            pairing_id=job.get('pairing_id', ''),
            condition=job.get('condition', ''),
            pair_bin=job.get('pair_bin', ''),
            selection_method=job.get('selection_method', ''),
            repeat_slot=job.get('repeat_slot', ''),
            pair_number=job.get('pair_number', ''),
            pair_key=job.get('pair_key', ''),
            target_pair_left_index=job.get('target_pair_left_index', ''),
            target_pair_right_index=job.get('target_pair_right_index', ''),
            target_class_name=job.get('shared_target_class_name', ''),
            poison_class_name=job.get('shared_poison_class_name', ''),
            feature_cosine_distance=job.get('feature_cosine_distance', ''),
            feature_cosine_similarity=job.get('feature_cosine_similarity', ''),
            gradient_cosine=job.get('gradient_cosine', ''),
            cluster_id=job.get('cluster_id', ''),
            cluster_size=job.get('cluster_size', ''),
            attacker_a_id=left_id,
            attacker_b_id=right_id,
            asr_a_alone=asr_a_alone,
            asr_b_alone=asr_b_alone,
            asr_a_combined=asr_a_combined,
            asr_b_combined=asr_b_combined,
            lift_a=lift_a,
            lift_b=lift_b,
            lift=0.5 * (lift_a + lift_b),
            quadrant=_quadrant(lift_a, lift_b),
        ))

    return summaries


def write_summary_csv(path, rows):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--repo-root', default='.', type=str)
    parser.add_argument('--output', default=None, type=str)
    args = parser.parse_args()

    experiment_path = os.path.expanduser(args.experiment)
    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    experiment = load_experiment(experiment_path)
    output = args.output
    if output is None:
        base, _ = os.path.splitext(experiment_path)
        output = f'{base}.summary.csv'

    rows = summarize_c6_experiment(experiment, repo_root)
    write_summary_csv(os.path.expanduser(output), rows)
    print(f'Wrote {len(rows)} C6 summary rows to {output}.')
