"""Experiment planners for dual-attacker studies."""

from __future__ import annotations

import os
import random

from .experiment import SCHEMA_VERSION


def _class_index(class_names, target_class):
    if isinstance(target_class, int):
        return target_class
    if isinstance(target_class, str) and target_class.isdigit():
        return int(target_class)
    if target_class not in class_names:
        raise ValueError(f'Unknown class {target_class}.')
    return class_names.index(target_class)


def _source_target_ranks(target_class_name, rankings):
    ranked = rankings[target_class_name]
    if len(ranked) < 9:
        raise ValueError('C1 planning expects 9 non-target classes.')
    return {int(entry['class_index']): rank + 1 for rank, entry in enumerate(ranked)}


def _fixed_source_pairs(target_class_idx, fixed_source_class_idx, class_names, distance_matrix):
    source_classes = [class_idx for class_idx in range(len(class_names)) if class_idx != target_class_idx]
    if fixed_source_class_idx == target_class_idx:
        raise ValueError('The fixed attacker-A source class must differ from the shared target class.')
    if fixed_source_class_idx not in source_classes:
        raise ValueError(f'Fixed attacker-A source class {fixed_source_class_idx} is invalid for target {target_class_idx}.')

    return [
        dict(
            left_class=fixed_source_class_idx,
            left_class_name=class_names[fixed_source_class_idx],
            right_class=source_class_idx,
            right_class_name=class_names[source_class_idx],
            source_source_distance=float(distance_matrix[fixed_source_class_idx, source_class_idx]),
        )
        for source_class_idx in source_classes
    ]


def _selection_keys(repeats):
    return [(2 * repeat_idx, 2 * repeat_idx + 1) for repeat_idx in range(repeats)]


def _sample_target_indices(class_to_valid_indices, source_classes, sample_count, planning_seed):
    sampled = {}
    for source_class_idx in sorted(source_classes):
        candidates = list(class_to_valid_indices[source_class_idx])
        if sample_count > len(candidates):
            raise ValueError(
                f'Need {sample_count} target images for class {source_class_idx}, but only found {len(candidates)}.'
            )
        rng = random.Random((planning_seed + 1) * 1_000_003 + source_class_idx)
        sampled[source_class_idx] = rng.sample(candidates, sample_count)
    return sampled


def _attacker_id(source_class_idx, selection_key, target_class_idx):
    return f'src{source_class_idx}_sel{selection_key}_to{target_class_idx}'


def build_c1_experiment(
    *,
    experiment_id,
    class_names,
    rankings,
    distance_matrix,
    class_to_valid_indices,
    shared_target_class,
    fixed_attacker_a_source_class,
    repeats,
    planning_seed,
    victim_seeds,
    common_args,
    output_root,
    scheduler=None,
    overlap_seed_base=0,
    distance_artifact_path=None,
):
    """Build a concrete C1 experiment JSON payload."""
    target_class_idx = _class_index(class_names, shared_target_class)
    target_class_name = class_names[target_class_idx]
    fixed_attacker_a_source_class_idx = _class_index(class_names, fixed_attacker_a_source_class)
    selection_keys = _selection_keys(repeats)
    source_target_ranks = _source_target_ranks(target_class_name, rankings)
    sampled_pairs = _fixed_source_pairs(
        target_class_idx,
        fixed_attacker_a_source_class_idx,
        class_names,
        distance_matrix,
    )
    source_classes = {int(pair['left_class']) for pair in sampled_pairs} | {int(pair['right_class']) for pair in sampled_pairs}
    sampled_target_indices = _sample_target_indices(class_to_valid_indices, source_classes, 2 * repeats, planning_seed)

    experiment_root = output_root
    brew_dir = f'{experiment_root}/brews'
    solo_dir = f'{experiment_root}/solo'
    dual_dir = f'{experiment_root}/dual'

    brew_jobs, solo_jobs, dual_jobs = [], [], []
    artifact_by_attacker = {}
    attacker_specs = {}

    for pair_idx, pair in enumerate(sampled_pairs):
        condition = f'fixed_{pair["left_class_name"]}_vs_{pair["right_class_name"]}'
        for repeat_idx, (left_key, right_key) in enumerate(selection_keys):
            attackers = []
            for side, source_class_idx, selection_key in (
                ('a', int(pair['left_class']), left_key),
                ('b', int(pair['right_class']), right_key),
            ):
                target_index = int(sampled_target_indices[source_class_idx][selection_key])
                attacker_key = (source_class_idx, selection_key, target_class_idx)
                if attacker_key not in attacker_specs:
                    attacker_id = _attacker_id(source_class_idx, selection_key, target_class_idx)
                    brew_job_id = f'brew_{attacker_id}'
                    artifact_path = os.path.join(brew_dir, f'{brew_job_id}.pt')
                    arg_overrides = dict(
                        poisonkey=f'{source_class_idx}-{target_class_idx}-{target_index}',
                        name=brew_job_id,
                        targets=1,
                    )
                    attacker_meta = dict(
                        attacker_id=attacker_id,
                        poisonkey=arg_overrides['poisonkey'],
                        brew_job_id=brew_job_id,
                        selection_key=selection_key,
                        target_index=target_index,
                        source_class=source_class_idx,
                        target_true_class=source_class_idx,
                        target_adv_class=target_class_idx,
                        source_target_distance=float(distance_matrix[target_class_idx, source_class_idx]),
                        source_target_rank=source_target_ranks[source_class_idx],
                    )
                    brew_jobs.append(dict(
                        job_id=brew_job_id,
                        attacker=attacker_meta,
                        arg_overrides=arg_overrides,
                        artifact_path=artifact_path,
                    ))
                    solo_jobs.append(dict(
                        job_id=f'solo_{attacker_id}',
                        attacker=attacker_meta,
                        brew_artifact_path=artifact_path,
                        victim_seeds=list(victim_seeds),
                        arg_overrides=dict(name=f'solo_{attacker_id}', poisonkey=None),
                        output_path=os.path.join(solo_dir, f'solo_{attacker_id}.csv'),
                    ))
                    artifact_by_attacker[attacker_id] = artifact_path
                    attacker_specs[attacker_key] = attacker_meta
                attackers.append(attacker_specs[attacker_key])

            pairing_id = f'{condition}_rep{repeat_idx + 1}'
            dual_jobs.append(dict(
                job_id=f'dual_{pairing_id}',
                pairing_id=pairing_id,
                condition=condition,
                distance_bucket='fixed_source_sweep',
                symmetric='',
                pair_distance_rank='',
                pair_distance_rank_fraction='',
                attackers=attackers,
                brew_artifact_paths=[artifact_by_attacker[attackers[0]['attacker_id']],
                                     artifact_by_attacker[attackers[1]['attacker_id']]],
                victim_seeds=list(victim_seeds),
                overlap_seed=overlap_seed_base + repeat_idx,
                overlap_policy='assign_one_owner',
                source_source_distance=pair['source_source_distance'],
                arg_overrides=dict(name=f'dual_{pairing_id}', poisonkey=None),
                output_path=os.path.join(dual_dir, f'dual_{pairing_id}.csv'),
            ))

    return dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        family='C1',
        class_names=list(class_names),
        metadata=dict(
            shared_target_class=target_class_idx,
            fixed_attacker_a_source_class=fixed_attacker_a_source_class_idx,
            planning_seed=planning_seed,
            repeats=repeats,
            sampled_pair_count=len(sampled_pairs),
            victim_seeds=list(victim_seeds),
            distance_artifact_path=distance_artifact_path,
            sampled_target_indices={str(class_idx): indices for class_idx, indices in sampled_target_indices.items()},
            pair_selection_strategy='fixed_attacker_a_source_sweep',
        ),
        scheduler=scheduler or {},
        common_args=common_args,
        output_root=experiment_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )
