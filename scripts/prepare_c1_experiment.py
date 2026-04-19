"""Prepare a concrete C1 dual-attacker experiment JSON."""

from __future__ import annotations

import argparse

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.config import c1_planner_defaults
from forest.dual_attack.prepare import (
    PreparationContext,
    add_shared_prepare_arguments,
    build_experiment_plan,
    compile_experiment_plan,
    load_preparation_context,
    materialize_experiment_plan,
    save_prepared_experiment_outputs,
)
from forest.dual_attack.planners import (
    _build_attacker_spec,
    _class_index,
    _fixed_source_pairs,
    _source_target_ranks,
)


def build_c1_plan(
    *,
    preparation_context,
    shared_target_class,
):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    victim_seeds = preparation_context.planner_defaults['victim_seeds']
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']
    target_class_idx = _class_index(class_names, shared_target_class)
    target_class_name = class_names[target_class_idx]
    fixed_attacker_a_source_class_idx = _class_index(
        class_names,
        preparation_context.fixed_attacker_a_source_class,
    )
    source_target_ranks = _source_target_ranks(target_class_name, preparation_context.rankings)
    sampled_pairs = _fixed_source_pairs(
        target_class_idx,
        fixed_attacker_a_source_class_idx,
        class_names,
        distance_matrix,
    )
    sampled_pairs = [
        pair for pair in sampled_pairs
        if int(pair['left_class']) != int(pair['right_class'])
    ]
    dual_requests = []

    for pair in sampled_pairs:
        condition = f'fixed_{pair["left_class_name"]}_vs_{pair["right_class_name"]}'
        for repeat_slot in range(repeats):
            attacker_requests = []
            for source_class_idx in (int(pair['left_class']), int(pair['right_class'])):
                attacker_requests.append(dict(
                    source_class_idx=source_class_idx,
                    target_class_idx=target_class_idx,
                    repeat_slot=repeat_slot,
                    source_target_distance=distance_matrix[target_class_idx, source_class_idx],
                    source_target_rank=source_target_ranks[source_class_idx],
                ))
            pairing_id = f'{condition}_rep{repeat_slot + 1}'
            dual_requests.append(dict(
                pairing_id=pairing_id,
                attacker_requests=attacker_requests,
                overlap_seed=overlap_seed_base + repeat_slot,
                metadata=dict(
                    condition=condition,
                    distance_bucket='fixed_source_sweep',
                    symmetric='',
                    pair_distance_rank='',
                    pair_distance_rank_fraction='',
                    source_source_distance=pair['source_source_distance'],
                ),
            ))

    return build_experiment_plan(
        family='C1',
        experiment_metadata=dict(
            shared_target_class=target_class_idx,
            fixed_attacker_a_source_class=fixed_attacker_a_source_class_idx,
            repeats=repeats,
            sampled_pair_count=len(sampled_pairs),
            victim_seeds=list(victim_seeds),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='fixed_attacker_a_source_sweep',
        ),
        dual_requests=dual_requests,
    )


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
    victim_seeds,
    common_args,
    output_root,
    scheduler=None,
    overlap_seed_base=0,
    distance_artifact_path=None,
):
    plan = build_c1_plan(
        preparation_context=PreparationContext(
            experiment_path=output_root,
            experiment_id=experiment_id,
            output_root=output_root,
            distance_artifact_path=distance_artifact_path,
            class_names=class_names,
            rankings=rankings,
            distance_matrix=distance_matrix,
            class_to_valid_indices=class_to_valid_indices,
            common_args=common_args,
            scheduler=scheduler or {},
            planner_defaults=dict(
                fixed_attacker_a_source_class=fixed_attacker_a_source_class,
                repeats=repeats,
                victim_seeds=list(victim_seeds),
                overlap_seed_base=overlap_seed_base,
            ),
            fixed_attacker_a_source_class=fixed_attacker_a_source_class,
            repeats=repeats,
        ),
        shared_target_class=shared_target_class,
    )
    return materialize_experiment_plan(
        experiment_id=experiment_id,
        class_names=class_names,
        plan=plan,
        class_to_valid_indices=class_to_valid_indices,
        victim_seeds=victim_seeds,
        common_args=common_args,
        output_root=output_root,
        build_attacker_spec=_build_attacker_spec,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_prepare_arguments(
        parser,
        fixed_attacker_help='Source class held fixed for attacker A while attacker B sweeps all non-target classes.',
    )
    parser.add_argument(
        '--shared_target_class',
        default=c1_planner_defaults()['shared_target_class'],
        type=str,
        help='Shared adversarial target class for the C1 sweep.',
    )
    args = parser.parse_args()
    preparation_context = load_preparation_context(args)
    experiment_plan = build_c1_plan(
        preparation_context=preparation_context,
        shared_target_class=args.shared_target_class,
    )
    experiment = compile_experiment_plan(
        preparation_context=preparation_context,
        experiment_plan=experiment_plan,
        build_attacker_spec=_build_attacker_spec,
    )
    save_prepared_experiment_outputs(
        preparation_context=preparation_context,
        experiment=experiment,
        experiment_plan=experiment_plan,
        save_plan_yaml_enabled=args.save_plan_yaml,
    )
    print(f'Saved C1 experiment spec to {preparation_context.experiment_path}.')
