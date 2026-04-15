"""Prepare a concrete C1 dual-attacker experiment JSON."""

from __future__ import annotations

import os

import torch

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

import forest
from forest.data.datasets import construct_datasets
from forest.dual_attack.experiment import save_experiment
from forest.dual_attack.planners import build_c1_experiment


PREP_OPTION_NAMES = {
    'distance_artifact',
    'output',
    'shared_target_class',
    'fixed_attacker_a_source_class',
    'planning_seed',
    'repeats',
    'victim_seeds',
    'overlap_seed_base',
    'slurm_account',
    'slurm_gpu',
    'slurm_mem',
    'slurm_cpus',
    'slurm_time',
}


def _parse_seed_list(raw_value):
    return [int(seed.strip()) for seed in raw_value.split(',') if seed.strip() != '']


def _class_to_valid_indices(dataset):
    mapping = {class_idx: [] for class_idx in range(len(dataset.classes))}
    for dataset_index in range(len(dataset)):
        label, item_index = dataset.get_target(dataset_index)
        mapping[int(label)].append(int(item_index))
    return mapping

if __name__ == "__main__":
    parser = forest.options()
    parser.set_defaults(randomize_deterministic_poison_ids=True)
    parser.add_argument('--distance_artifact', required=True, type=str, help='Saved class-distance matrix artifact.')
    parser.add_argument('--output', required=True, type=str, help='Where to save the experiment JSON.')
    parser.add_argument('--shared_target_class', default='airplane', type=str,
                        help='Shared adversarial target class for the C1 sweep.')
    parser.add_argument('--fixed_attacker_a_source_class', default='dog', type=str,
                        help='Source class held fixed for attacker A while attacker B sweeps all non-target classes.')
    parser.add_argument('--planning_seed', default=0, type=int,
                        help='Random seed used once during planning to choose target images before freezing the JSON.')
    parser.add_argument('--repeats', default=3, type=int, help='Number of repeat pairings per condition.')
    parser.add_argument('--victim_seeds', default='0,1,2,3,4,5,6,7', type=str,
                        help='Comma-separated victim seeds shared by solo and dual runs.')
    parser.add_argument('--overlap_seed_base', default=0, type=int, help='Base seed for deterministic overlap resolution.')
    parser.add_argument('--slurm_account', default=None, type=str)
    parser.add_argument('--slurm_gpu', default='l40s', type=str)
    parser.add_argument('--slurm_mem', default='20G', type=str)
    parser.add_argument('--slurm_cpus', default=4, type=int)
    parser.add_argument('--slurm_time', default='4:00:00', type=str)
    args = parser.parse_args()

    distance_artifact = torch.load(os.path.expanduser(args.distance_artifact), map_location='cpu')
    _, validset = construct_datasets(args.dataset, args.data_path, normalize=False)
    class_to_valid_indices = _class_to_valid_indices(validset)

    experiment_path = os.path.expanduser(args.output)
    experiment_id = os.path.splitext(os.path.basename(experiment_path))[0]
    output_root = os.path.join(os.path.dirname(experiment_path), experiment_id)

    common_args = {key: value for key, value in vars(args).items() if key not in PREP_OPTION_NAMES}
    scheduler = dict(
        account=args.slurm_account,
        gpu=args.slurm_gpu,
        mem=args.slurm_mem,
        cpus=args.slurm_cpus,
        time=args.slurm_time,
    )

    experiment = build_c1_experiment(
        experiment_id=experiment_id,
        class_names=distance_artifact['class_names'],
        rankings=distance_artifact['rankings'],
        distance_matrix=distance_artifact['distance_matrix'],
        class_to_valid_indices=class_to_valid_indices,
        shared_target_class=args.shared_target_class,
        fixed_attacker_a_source_class=args.fixed_attacker_a_source_class,
        planning_seed=args.planning_seed,
        repeats=args.repeats,
        victim_seeds=_parse_seed_list(args.victim_seeds),
        common_args=common_args,
        output_root=output_root,
        scheduler=scheduler,
        overlap_seed_base=args.overlap_seed_base,
        distance_artifact_path=os.path.expanduser(args.distance_artifact),
    )
    save_experiment(experiment, experiment_path)
    print(f'Saved C1 experiment spec to {experiment_path}.')
