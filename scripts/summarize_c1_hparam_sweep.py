"""Summarize C1 eps/budget sweep CSV outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics


def _float(row, key, default=0.0):
    value = row.get(key, '')
    if value == '':
        return default
    return float(value)


def _group_key(row):
    return (
        row.get('eps', ''),
        row.get('budget', ''),
        row.get('source_class_name', ''),
        row.get('target_adv_class_name', ''),
        row.get('target_index', ''),
    )


def _read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row['_path'] = path
                rows.append(row)
    return rows


def _summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    summaries = []
    for (eps, budget, source_name, target_name, target_index), group_rows in groups.items():
        successes = [_float(row, 'success') for row in group_rows]
        confidences = [_float(row, 'adv_confidence') for row in group_rows]
        true_confidences = [_float(row, 'true_confidence') for row in group_rows]
        losses = [_float(row, 'target_loss') for row in group_rows]
        summaries.append(dict(
            eps=float(eps),
            budget=float(budget),
            source_class_name=source_name,
            target_adv_class_name=target_name,
            target_index=int(target_index),
            runs=len(group_rows),
            success_rate=sum(successes) / len(successes),
            mean_adv_confidence=statistics.fmean(confidences),
            median_adv_confidence=statistics.median(confidences),
            min_adv_confidence=min(confidences),
            max_adv_confidence=max(confidences),
            mean_true_confidence=statistics.fmean(true_confidences),
            mean_target_loss=statistics.fmean(losses),
        ))
    return sorted(summaries, key=lambda row: (-row['eps'], -row['budget'], row['target_index']))


def _write_csv(path, summaries):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _print_table(summaries):
    header = (
        'eps',
        'budget',
        'target_index',
        'runs',
        'success_rate',
        'mean_adv_conf',
        'median_adv_conf',
        'mean_true_conf',
    )
    print(','.join(header))
    for row in summaries:
        print(
            f'{row["eps"]:g},'
            f'{row["budget"]:g},'
            f'{row["target_index"]},'
            f'{row["runs"]},'
            f'{row["success_rate"]:.3f},'
            f'{row["mean_adv_confidence"]:.4f},'
            f'{row["median_adv_confidence"]:.4f},'
            f'{row["mean_true_confidence"]:.4f}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize C1 hparam sweep solo CSVs.')
    parser.add_argument('--input-glob', default='artifacts/c1/c1_dog_to_airplane_hparam_sweep/solo/*.csv', type=str)
    parser.add_argument('--output', default=None, type=str, help='Optional CSV summary output path.')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.expanduser(args.input_glob)))
    if len(paths) == 0:
        raise SystemExit(f'No CSV files matched {args.input_glob}.')

    summaries = _summarize(_read_rows(paths))
    _print_table(summaries)
    if args.output is not None:
        _write_csv(os.path.expanduser(args.output), summaries)
        print(f'Wrote summary to {args.output}.')
