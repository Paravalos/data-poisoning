"""Prepare a concrete C5 overlap-collision experiment JSON derived from a C1 spec."""

from __future__ import annotations

import argparse
import os

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.data.datasets import construct_datasets
from forest.dual_attack.experiment import load_experiment, save_experiment
from forest.dual_attack.planners import build_c5_experiment


def _parse_seed_list(raw_value):
    return [int(value.strip()) for value in raw_value.split(',') if value.strip() != '']


def _class_to_train_indices(dataset):
    mapping = {class_idx: [] for class_idx in range(len(dataset.classes))}
    for dataset_index in range(len(dataset)):
        label, item_index = dataset.get_target(dataset_index)
        mapping[int(label)].append(int(item_index))
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c1_experiment', required=True, type=str, help='Path to the source C1 experiment JSON.')
    parser.add_argument('--output', required=True, type=str, help='Where to save the derived C5 experiment JSON.')
    parser.add_argument('--pair_condition', default='fixed_dog_vs_cat', type=str,
                        help='C1 condition to derive the C5 overlap sweep from.')
    parser.add_argument('--overlap_levels', default='0,25,50,100', type=str,
                        help='Comma-separated overlap percentages for the C5 sweep.')
    parser.add_argument('--merge_rule', default='sum_clipped', choices=['sum_clipped'], type=str,
                        help='How to combine perturbations on shared poison images.')
    parser.add_argument('--slurm_account', default=None, type=str)
    parser.add_argument('--slurm_gpu', default=None, type=str)
    parser.add_argument('--slurm_mem', default=None, type=str)
    parser.add_argument('--slurm_cpus', default=None, type=int)
    parser.add_argument('--slurm_time', default=None, type=str)
    args = parser.parse_args()

    c1_experiment_path = os.path.expanduser(args.c1_experiment)
    c1_experiment = load_experiment(c1_experiment_path)
    common_args = c1_experiment.get('common_args', {})
    if 'dataset' not in common_args or 'data_path' not in common_args:
        raise ValueError('The source C1 experiment must define dataset and data_path in common_args.')

    trainset, _ = construct_datasets(common_args['dataset'], common_args['data_path'], normalize=False)
    class_to_train_indices = _class_to_train_indices(trainset)

    experiment_path = os.path.expanduser(args.output)
    experiment_id = os.path.splitext(os.path.basename(experiment_path))[0]
    output_root = os.path.join(os.path.dirname(experiment_path), experiment_id)

    scheduler = dict(c1_experiment.get('scheduler', {}))
    if args.slurm_account is not None:
        scheduler['account'] = args.slurm_account
    if args.slurm_gpu is not None:
        scheduler['gpu'] = args.slurm_gpu
    if args.slurm_mem is not None:
        scheduler['mem'] = args.slurm_mem
    if args.slurm_cpus is not None:
        scheduler['cpus'] = args.slurm_cpus
    if args.slurm_time is not None:
        scheduler['time'] = args.slurm_time

    experiment = build_c5_experiment(
        experiment_id=experiment_id,
        c1_experiment=c1_experiment,
        class_to_train_indices=class_to_train_indices,
        pair_condition=args.pair_condition,
        overlap_percentages=_parse_seed_list(args.overlap_levels),
        output_root=output_root,
        scheduler=scheduler,
        source_experiment_path=c1_experiment_path,
        merge_rule=args.merge_rule,
    )
    save_experiment(experiment, experiment_path)
    print(f'Saved C5 experiment spec to {experiment_path}.')
