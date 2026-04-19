"""Prepare a concrete C2 dual-attacker experiment JSON."""

from __future__ import annotations

import argparse

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

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
    C2_MOTIF_ORDER,
    _build_attacker_spec,
    _class_index,
    _pair_overlap_seed,
    _select_c2_motif_targets,
    _source_strata_pairs,
    _source_target_ranks,
)


def build_c2_plan(
    *,
    preparation_context,
):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    victim_seeds = preparation_context.planner_defaults['victim_seeds']
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']
    fixed_attacker_a_source_class_idx = _class_index(
        class_names,
        preparation_context.fixed_attacker_a_source_class,
    )
    source_pairs = _source_strata_pairs(fixed_attacker_a_source_class_idx, class_names, distance_matrix)
    dual_requests = []
    source_target_ranks_by_target = {}
    motif_specs_by_pair = {}

    for source_pair in source_pairs:
        motif_specs_by_pair[source_pair['source_pair_label']] = _select_c2_motif_targets(
            source_pair,
            class_names,
            distance_matrix,
        )

    for source_pair in source_pairs:
        motif_specs = motif_specs_by_pair[source_pair['source_pair_label']]
        for motif_label in C2_MOTIF_ORDER:
            motif_spec = motif_specs[motif_label]
            condition = f'{source_pair["source_pair_label"]}_{motif_label}'
            for repeat_slot in range(repeats):
                attacker_requests = []
                for source_class_idx, target_class_idx in (
                    (int(source_pair['left_class']), int(motif_spec['target_a_class'])),
                    (int(source_pair['right_class']), int(motif_spec['target_b_class'])),
                ):
                    if target_class_idx not in source_target_ranks_by_target:
                        source_target_ranks_by_target[target_class_idx] = _source_target_ranks(
                            class_names[target_class_idx],
                            preparation_context.rankings,
                        )
                    attacker_requests.append(dict(
                        source_class_idx=source_class_idx,
                        target_class_idx=target_class_idx,
                        repeat_slot=repeat_slot,
                        source_target_distance=distance_matrix[target_class_idx, source_class_idx],
                        source_target_rank=source_target_ranks_by_target[target_class_idx][source_class_idx],
                    ))
                pairing_id = f'{condition}_rep{repeat_slot + 1}'
                dual_requests.append(dict(
                    pairing_id=pairing_id,
                    attacker_requests=attacker_requests,
                    overlap_seed=_pair_overlap_seed(pairing_id, overlap_seed_base),
                    metadata=dict(
                        condition=condition,
                        distance_bucket=source_pair['source_stratum'],
                        symmetric='',
                        pair_distance_rank='',
                        pair_distance_rank_fraction='',
                        source_pair_label=source_pair['source_pair_label'],
                        source_stratum=source_pair['source_stratum'],
                        source_source_rank=source_pair['source_source_rank'],
                        motif_label=motif_label,
                        motif_rank=motif_spec['motif_rank'],
                        alignment_type=motif_spec['alignment_type'],
                        target_a_class=int(motif_spec['target_a_class']),
                        target_a_class_name=motif_spec['target_a_class_name'],
                        target_b_class=int(motif_spec['target_b_class']),
                        target_b_class_name=motif_spec['target_b_class_name'],
                        S=float(source_pair['source_source_distance']),
                        T=float(motif_spec['target_target_distance']),
                        G=float(motif_spec['cross_alignment_gap']),
                        a_self=float(motif_spec['a_self']),
                        b_self=float(motif_spec['b_self']),
                        a_cross=float(motif_spec['a_cross']),
                        b_cross=float(motif_spec['b_cross']),
                        source_source_distance=float(source_pair['source_source_distance']),
                        target_target_distance=float(motif_spec['target_target_distance']),
                        cross_alignment_gap=float(motif_spec['cross_alignment_gap']),
                    ),
                ))

    return build_experiment_plan(
        family='C2',
        experiment_metadata=dict(
            fixed_attacker_a_source_class=fixed_attacker_a_source_class_idx,
            repeats=repeats,
            source_strata=[
                dict(
                    source_stratum=pair['source_stratum'],
                    partner_source_class=pair['right_class'],
                    partner_source_class_name=pair['right_class_name'],
                    source_source_distance=pair['source_source_distance'],
                    source_source_rank=pair['source_source_rank'],
                )
                for pair in source_pairs
            ],
            motif_labels=list(C2_MOTIF_ORDER),
            victim_seeds=list(victim_seeds),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='fixed_attacker_a_source_strata_target_sweep',
        ),
        dual_requests=dual_requests,
    )


def build_c2_experiment(
    *,
    experiment_id,
    class_names,
    rankings,
    distance_matrix,
    class_to_valid_indices,
    fixed_attacker_a_source_class,
    repeats,
    victim_seeds,
    common_args,
    output_root,
    scheduler=None,
    overlap_seed_base=0,
    distance_artifact_path=None,
):
    plan = build_c2_plan(
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
        fixed_attacker_help='Source class held fixed for attacker A while C2 sweeps target geometry.',
    )
    args = parser.parse_args()
    preparation_context = load_preparation_context(args)
    experiment_plan = build_c2_plan(
        preparation_context=preparation_context,
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
    print(f'Saved C2 experiment spec to {preparation_context.experiment_path}.')
