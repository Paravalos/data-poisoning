"""Shared helpers for dual-attack experiment preparation scripts."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

from .config import common_args_defaults, planner_defaults, scheduler_defaults
from .experiment import SCHEMA_VERSION


@dataclass(frozen=True)
class PreparationContext:
    experiment_path: str
    experiment_id: str
    output_root: str
    distance_artifact_path: str
    class_names: list
    rankings: dict
    distance_matrix: object
    class_to_valid_indices: dict
    common_args: dict
    scheduler: dict
    planner_defaults: dict
    fixed_attacker_a_source_class: str
    repeats: int


def class_to_dataset_indices(dataset):
    mapping = {class_idx: [] for class_idx in range(len(dataset.classes))}
    for dataset_index in range(len(dataset)):
        label, item_index = dataset.get_target(dataset_index)
        mapping[int(label)].append(int(item_index))
    return mapping


def add_shared_prepare_arguments(parser, *, fixed_attacker_help):
    shared_planner_defaults = planner_defaults()
    parser.add_argument('--distance_artifact', required=True, type=str, help='Saved class-distance matrix artifact.')
    parser.add_argument('--output', required=True, type=str, help='Where to save the experiment JSON.')
    parser.add_argument(
        '--save_plan_yaml',
        action='store_true',
        help='Also save a readable YAML plan next to the experiment JSON.',
    )
    parser.add_argument(
        '--fixed_attacker_a_source_class',
        default=shared_planner_defaults['fixed_attacker_a_source_class'],
        type=str,
        help=fixed_attacker_help,
    )
    parser.add_argument(
        '--repeats',
        default=shared_planner_defaults['repeats'],
        type=int,
        help='Number of repeat pairings per condition.',
    )


def output_paths_from_output_arg(raw_output):
    experiment_path = os.path.expanduser(raw_output)
    experiment_id = os.path.splitext(os.path.basename(experiment_path))[0]
    output_root = os.path.join(os.path.dirname(experiment_path), experiment_id)
    return experiment_path, experiment_id, output_root


def plan_yaml_path(experiment_path):
    base, _ = os.path.splitext(os.path.expanduser(experiment_path))
    return f'{base}.plan.yaml'


def common_args_from_config():
    return common_args_defaults()


def scheduler_from_config():
    return scheduler_defaults()


def load_preparation_context(args):
    import torch

    from forest.data.datasets import construct_datasets

    common_args = common_args_from_config()
    scheduler = scheduler_from_config()
    shared_planner_defaults = planner_defaults()
    distance_artifact_path = os.path.expanduser(args.distance_artifact)
    distance_artifact = torch.load(distance_artifact_path, map_location='cpu')
    _, validset = construct_datasets(common_args['dataset'], common_args['data_path'], normalize=False)
    class_to_valid_indices = class_to_dataset_indices(validset)
    experiment_path, experiment_id, output_root = output_paths_from_output_arg(args.output)
    return PreparationContext(
        experiment_path=experiment_path,
        experiment_id=experiment_id,
        output_root=output_root,
        distance_artifact_path=distance_artifact_path,
        class_names=list(distance_artifact['class_names']),
        rankings=distance_artifact['rankings'],
        distance_matrix=distance_artifact['distance_matrix'],
        class_to_valid_indices=class_to_valid_indices,
        common_args=common_args,
        scheduler=scheduler,
        planner_defaults=shared_planner_defaults,
        fixed_attacker_a_source_class=args.fixed_attacker_a_source_class,
        repeats=args.repeats,
    )


def stage_directories(output_root):
    return dict(
        brew=os.path.join(output_root, 'brews'),
        solo=os.path.join(output_root, 'solo'),
        dual=os.path.join(output_root, 'dual'),
    )


def _stable_int_seed(value):
    return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)


def _target_permutation(class_to_valid_indices, source_class_idx, target_class_idx):
    candidates = list(class_to_valid_indices[int(source_class_idx)])
    rng = random.Random(_stable_int_seed(f'target:{int(source_class_idx)}->{int(target_class_idx)}'))
    rng.shuffle(candidates)
    return candidates


def assign_target_indices(
    *,
    attacker_requests,
    class_to_valid_indices,
):
    resolved_requests = [dict(request) for request in attacker_requests]
    permutations = {}
    seen_attacker_keys = set()

    for request in resolved_requests:
        source_class_idx = int(request['source_class_idx'])
        target_class_idx = int(request['target_class_idx'])
        repeat_slot = int(request['repeat_slot'])
        identity_key = (source_class_idx, target_class_idx, repeat_slot)
        if identity_key in seen_attacker_keys:
            raise ValueError(
                'Dual jobs cannot contain the same (source_class, target_class, repeat_slot) attacker twice.'
            )
        seen_attacker_keys.add(identity_key)

        permutation_key = (source_class_idx, target_class_idx)
        if permutation_key not in permutations:
            permutations[permutation_key] = _target_permutation(
                class_to_valid_indices,
                source_class_idx,
                target_class_idx,
            )

        candidates = permutations[permutation_key]
        if repeat_slot >= len(candidates):
            raise ValueError(
                f'Need repeat slot {repeat_slot} for attacker {source_class_idx}->{target_class_idx}, '
                f'but only found {len(candidates)} candidate target images.'
            )

        target_index = int(candidates[repeat_slot])
        request['target_index'] = target_index
        request['attacker_key'] = (
            source_class_idx,
            repeat_slot,
            target_class_idx,
            target_index,
        )

    return resolved_requests


def register_attacker_jobs(
    *,
    attacker_key,
    attacker_spec,
    attacker_specs,
    artifact_by_attacker,
    brew_jobs,
    solo_jobs,
    solo_dir,
    victim_seeds,
):
    if attacker_key not in attacker_specs:
        brew_jobs.append(dict(
            job_id=attacker_spec['job_id'],
            attacker=attacker_spec['attacker'],
            arg_overrides=attacker_spec['arg_overrides'],
            artifact_path=attacker_spec['artifact_path'],
        ))
        solo_jobs.append(dict(
            job_id=f'solo_{attacker_spec["attacker"]["attacker_id"]}',
            attacker=attacker_spec['attacker'],
            brew_artifact_path=attacker_spec['artifact_path'],
            victim_seeds=list(victim_seeds),
            arg_overrides=dict(name=f'solo_{attacker_spec["attacker"]["attacker_id"]}', poisonkey=None),
            output_path=os.path.join(solo_dir, f'solo_{attacker_spec["attacker"]["attacker_id"]}.csv'),
        ))
        artifact_by_attacker[attacker_spec['attacker']['attacker_id']] = attacker_spec['artifact_path']
        attacker_specs[attacker_key] = attacker_spec['attacker']
    return attacker_specs[attacker_key]


def ensure_registered_attacker(
    *,
    attacker_key,
    build_attacker_spec,
    attacker_specs,
    artifact_by_attacker,
    brew_jobs,
    solo_jobs,
    solo_dir,
    victim_seeds,
    **attacker_spec_kwargs,
):
    if attacker_key in attacker_specs:
        return attacker_specs[attacker_key]

    attacker_spec = build_attacker_spec(**attacker_spec_kwargs)
    return register_attacker_jobs(
        attacker_key=attacker_key,
        attacker_spec=attacker_spec,
        attacker_specs=attacker_specs,
        artifact_by_attacker=artifact_by_attacker,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        solo_dir=solo_dir,
        victim_seeds=victim_seeds,
    )


def materialize_attackers(
    *,
    attacker_requests,
    build_attacker_spec,
    attacker_specs,
    artifact_by_attacker,
    brew_jobs,
    solo_jobs,
    solo_dir,
    victim_seeds,
):
    attackers = []
    for request in attacker_requests:
        request = dict(request)
        attacker_key = request.pop('attacker_key')
        attackers.append(ensure_registered_attacker(
            attacker_key=attacker_key,
            build_attacker_spec=build_attacker_spec,
            attacker_specs=attacker_specs,
            artifact_by_attacker=artifact_by_attacker,
            brew_jobs=brew_jobs,
            solo_jobs=solo_jobs,
            solo_dir=solo_dir,
            victim_seeds=victim_seeds,
            **request,
        ))
    return attackers


def build_dual_job(
    *,
    pairing_id,
    attackers,
    artifact_by_attacker,
    victim_seeds,
    overlap_seed,
    dual_dir,
    extra_fields=None,
):
    return dict(
        job_id=f'dual_{pairing_id}',
        pairing_id=pairing_id,
        attackers=attackers,
        brew_artifact_paths=[artifact_by_attacker[attacker['attacker_id']] for attacker in attackers],
        victim_seeds=list(victim_seeds),
        overlap_seed=overlap_seed,
        overlap_policy='assign_one_owner',
        arg_overrides=dict(name=f'dual_{pairing_id}', poisonkey=None),
        output_path=os.path.join(dual_dir, f'dual_{pairing_id}.csv'),
        **(extra_fields or {}),
    )


def build_experiment_plan(
    *,
    family,
    experiment_metadata,
    dual_requests,
):
    return dict(
        family=family,
        experiment_metadata=experiment_metadata,
        dual_requests=dual_requests,
    )


def materialize_experiment_plan(
    *,
    experiment_id,
    class_names,
    plan,
    class_to_valid_indices,
    victim_seeds,
    common_args,
    output_root,
    build_attacker_spec,
    scheduler=None,
):
    directories = stage_directories(output_root)
    brew_jobs, solo_jobs, dual_jobs = [], [], []
    artifact_by_attacker = {}
    attacker_specs = {}

    for dual_request in plan['dual_requests']:
        attacker_requests = [
            dict(attacker_request, brew_dir=attacker_request.get('brew_dir', directories['brew']))
            for attacker_request in dual_request['attacker_requests']
        ]
        attackers = materialize_attackers(
            attacker_requests=assign_target_indices(
                attacker_requests=attacker_requests,
                class_to_valid_indices=class_to_valid_indices,
            ),
            build_attacker_spec=build_attacker_spec,
            attacker_specs=attacker_specs,
            artifact_by_attacker=artifact_by_attacker,
            brew_jobs=brew_jobs,
            solo_jobs=solo_jobs,
            solo_dir=directories['solo'],
            victim_seeds=victim_seeds,
        )
        dual_jobs.append(build_dual_job(
            pairing_id=dual_request['pairing_id'],
            attackers=attackers,
            artifact_by_attacker=artifact_by_attacker,
            victim_seeds=victim_seeds,
            overlap_seed=dual_request['overlap_seed'],
            dual_dir=directories['dual'],
            extra_fields=dual_request.get('metadata'),
        ))

    return build_experiment_manifest(
        experiment_id=experiment_id,
        family=plan['family'],
        class_names=class_names,
        metadata=plan['experiment_metadata'],
        scheduler=scheduler,
        common_args=common_args,
        output_root=output_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )


def compile_experiment_plan(
    *,
    preparation_context,
    experiment_plan,
    build_attacker_spec,
):
    return materialize_experiment_plan(
        experiment_id=preparation_context.experiment_id,
        class_names=preparation_context.class_names,
        plan=experiment_plan,
        class_to_valid_indices=preparation_context.class_to_valid_indices,
        victim_seeds=preparation_context.planner_defaults['victim_seeds'],
        common_args=preparation_context.common_args,
        output_root=preparation_context.output_root,
        build_attacker_spec=build_attacker_spec,
        scheduler=preparation_context.scheduler,
    )


def _render_plan_attacker(attacker_request, class_names):
    return dict(
        source_class=int(attacker_request['source_class_idx']),
        source_class_name=class_names[int(attacker_request['source_class_idx'])],
        target_class=int(attacker_request['target_class_idx']),
        target_class_name=class_names[int(attacker_request['target_class_idx'])],
        repeat_slot=int(attacker_request['repeat_slot']),
    )


def save_plan_yaml(plan, class_names, path):
    import yaml

    payload = dict(
        family=plan['family'],
        experiment_metadata=plan['experiment_metadata'],
        dual_requests=[
            dict(
                pairing_id=dual_request['pairing_id'],
                overlap_seed=dual_request['overlap_seed'],
                attackers=[
                    _render_plan_attacker(attacker_request, class_names)
                    for attacker_request in dual_request['attacker_requests']
                ],
                metadata=dual_request.get('metadata', {}),
            )
            for dual_request in plan['dual_requests']
        ],
    )
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_prepared_experiment_outputs(
    *,
    preparation_context,
    experiment,
    experiment_plan,
    save_plan_yaml_enabled=False,
):
    from .experiment import save_experiment

    save_experiment(experiment, preparation_context.experiment_path)
    if save_plan_yaml_enabled:
        save_plan_yaml(
            experiment_plan,
            preparation_context.class_names,
            plan_yaml_path(preparation_context.experiment_path),
        )


def build_experiment_manifest(
    *,
    experiment_id,
    family,
    class_names,
    metadata,
    scheduler,
    common_args,
    output_root,
    brew_jobs,
    solo_jobs,
    dual_jobs,
):
    return dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        family=family,
        class_names=list(class_names),
        metadata=metadata,
        scheduler=scheduler or {},
        common_args=common_args,
        output_root=output_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )
