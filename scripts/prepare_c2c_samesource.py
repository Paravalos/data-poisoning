"""Prepare a C2c same-source experiment: clones (A=B) and forks (same source, different targets).

C1 covers different-source-same-target; C2 covers different-source-different-target. C2c
fills in the same-source row of the 2x2 (source x target) design space:

  - clone condition   : both attackers share (source, target) but run on disjoint repeat
                        slots so they have independent target-image draws and
                        independent poison-id seeds. Measures the cooperation ceiling
                        at fixed (source, target): if two copies can't cooperate, no
                        partner-geometry change will make them.
  - fork conditions   : both attackers share a source class but pick different targets.
                        Tests whether two attackers pulling poisons from the same class
                        but aiming elsewhere interfere or cooperate. Two fork flavors:
                        both-easy targets (close to source) and both-hard (far).

For each of SOURCE_CLASSES: 1 clone + 2 forks = 3 conditions x repeats reps.
Default 4 sources x 3 conditions x 8 reps = 96 dual jobs.
"""

from __future__ import annotations

import argparse

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.prepare import (
    build_experiment_plan,
    compile_experiment_plan,
    load_preparation_context,
    save_prepared_experiment_outputs,
    add_shared_prepare_arguments,
)
from forest.dual_attack.planners import (
    _build_attacker_spec,
    _class_index,
    _pair_overlap_seed,
    _source_target_ranks,
)


# Source classes chosen to span CIFAR-10 geometry (two animals, two vehicles, mixed difficulty).
SOURCE_CLASSES = ('dog', 'airplane', 'frog', 'truck')

CONDITION_ORDER = ('clone', 'fork_easy', 'fork_hard')


def _targets_ranked_by_source_distance(source_class_idx, class_names, distance_matrix):
    """Non-source classes ordered by d(source, class), closest first."""
    return sorted(
        (class_idx for class_idx in range(len(class_names)) if class_idx != source_class_idx),
        key=lambda class_idx: float(distance_matrix[source_class_idx, class_idx]),
    )


def _pick_targets_for_source(source_class_idx, class_names, distance_matrix):
    """Pick five targets spanning difficulty: two easy, one mid (for clone), two hard."""
    ranked = _targets_ranked_by_source_distance(source_class_idx, class_names, distance_matrix)
    n = len(ranked)
    return dict(
        easy_a=ranked[0],
        easy_b=ranked[1],
        mid=ranked[n // 2],
        hard_a=ranked[-2],
        hard_b=ranked[-1],
    )


def _ensure_source_target_ranks(target_class_idx, rankings, class_names, cache):
    if target_class_idx not in cache:
        cache[target_class_idx] = _source_target_ranks(class_names[target_class_idx], rankings)
    return cache[target_class_idx]


def _attacker_request(
    *,
    source_class_idx,
    target_class_idx,
    repeat_slot,
    distance_matrix,
    source_target_ranks_by_target,
    rankings,
    class_names,
):
    ranks = _ensure_source_target_ranks(
        target_class_idx, rankings, class_names, source_target_ranks_by_target,
    )
    return dict(
        source_class_idx=source_class_idx,
        target_class_idx=target_class_idx,
        repeat_slot=repeat_slot,
        source_target_distance=float(distance_matrix[target_class_idx, source_class_idx]),
        source_target_rank=ranks[source_class_idx],
    )


def build_c2c_plan(*, preparation_context, source_class_names):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']

    dual_requests = []
    source_summaries = []
    source_target_ranks_by_target = {}

    for source_class_name in source_class_names:
        source_class_idx = _class_index(class_names, source_class_name)
        targets = _pick_targets_for_source(source_class_idx, class_names, distance_matrix)

        # Clone: A uses slots 0..repeats-1, B uses slots repeats..2*repeats-1, same target.
        clone_target_idx = targets['mid']
        for repeat_slot in range(repeats):
            a_slot = repeat_slot
            b_slot = repeat_slot + repeats
            attacker_requests = [
                _attacker_request(
                    source_class_idx=source_class_idx,
                    target_class_idx=clone_target_idx,
                    repeat_slot=slot,
                    distance_matrix=distance_matrix,
                    source_target_ranks_by_target=source_target_ranks_by_target,
                    rankings=preparation_context.rankings,
                    class_names=class_names,
                )
                for slot in (a_slot, b_slot)
            ]
            pairing_id = f'{source_class_name}__clone_rep{repeat_slot + 1}'
            dual_requests.append(dict(
                pairing_id=pairing_id,
                attacker_requests=attacker_requests,
                overlap_seed=_pair_overlap_seed(pairing_id, overlap_seed_base),
                metadata=dict(
                    condition=f'{source_class_name}__clone',
                    distance_bucket='samesource_clone',
                    source_class=source_class_idx,
                    source_class_name=source_class_name,
                    condition_kind='clone',
                    target_a_class=int(clone_target_idx),
                    target_a_class_name=class_names[clone_target_idx],
                    target_b_class=int(clone_target_idx),
                    target_b_class_name=class_names[clone_target_idx],
                    a_self=float(distance_matrix[clone_target_idx, source_class_idx]),
                    b_self=float(distance_matrix[clone_target_idx, source_class_idx]),
                    target_target_distance=0.0,
                    a_repeat_slot=a_slot,
                    b_repeat_slot=b_slot,
                ),
            ))

        # Forks: both attackers share the source; A and B pick distinct targets from the same
        # difficulty tier. fork_easy uses the two closest targets; fork_hard uses the two farthest.
        for condition_kind, target_a_idx, target_b_idx in (
            ('fork_easy', targets['easy_a'], targets['easy_b']),
            ('fork_hard', targets['hard_a'], targets['hard_b']),
        ):
            for repeat_slot in range(repeats):
                attacker_requests = [
                    _attacker_request(
                        source_class_idx=source_class_idx,
                        target_class_idx=target_idx,
                        repeat_slot=repeat_slot,
                        distance_matrix=distance_matrix,
                        source_target_ranks_by_target=source_target_ranks_by_target,
                        rankings=preparation_context.rankings,
                        class_names=class_names,
                    )
                    for target_idx in (target_a_idx, target_b_idx)
                ]
                pairing_id = f'{source_class_name}__{condition_kind}_rep{repeat_slot + 1}'
                dual_requests.append(dict(
                    pairing_id=pairing_id,
                    attacker_requests=attacker_requests,
                    overlap_seed=_pair_overlap_seed(pairing_id, overlap_seed_base),
                    metadata=dict(
                        condition=f'{source_class_name}__{condition_kind}',
                        distance_bucket=f'samesource_{condition_kind}',
                        source_class=source_class_idx,
                        source_class_name=source_class_name,
                        condition_kind=condition_kind,
                        target_a_class=int(target_a_idx),
                        target_a_class_name=class_names[target_a_idx],
                        target_b_class=int(target_b_idx),
                        target_b_class_name=class_names[target_b_idx],
                        a_self=float(distance_matrix[target_a_idx, source_class_idx]),
                        b_self=float(distance_matrix[target_b_idx, source_class_idx]),
                        target_target_distance=float(distance_matrix[target_a_idx, target_b_idx]),
                    ),
                ))

        source_summaries.append(dict(
            source_class=source_class_idx,
            source_class_name=source_class_name,
            clone_target_class=int(clone_target_idx),
            clone_target_class_name=class_names[clone_target_idx],
            fork_easy_targets=[int(targets['easy_a']), int(targets['easy_b'])],
            fork_easy_target_names=[class_names[targets['easy_a']], class_names[targets['easy_b']]],
            fork_hard_targets=[int(targets['hard_a']), int(targets['hard_b'])],
            fork_hard_target_names=[class_names[targets['hard_a']], class_names[targets['hard_b']]],
        ))

    return build_experiment_plan(
        family='C2c',
        experiment_metadata=dict(
            repeats=repeats,
            source_classes=source_summaries,
            condition_order=list(CONDITION_ORDER),
            victim_seeds=list(preparation_context.planner_defaults['victim_seeds']),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='samesource_clone_and_fork',
        ),
        dual_requests=dual_requests,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_prepare_arguments(
        parser,
        fixed_attacker_help='Unused in C2c (source classes are enumerated from SOURCE_CLASSES).',
    )
    parser.add_argument(
        '--source_classes',
        nargs='+',
        default=list(SOURCE_CLASSES),
        help=f'Override the default source-class list {list(SOURCE_CLASSES)}.',
    )
    parser.set_defaults(repeats=8)
    args = parser.parse_args()
    preparation_context = load_preparation_context(args)
    experiment_plan = build_c2c_plan(
        preparation_context=preparation_context,
        source_class_names=tuple(args.source_classes),
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
    print(
        f'Saved C2c experiment spec to {preparation_context.experiment_path} '
        f'(duals={len(experiment["dual_jobs"])}, solos={len(experiment["solo_jobs"])}, '
        f'brews={len(experiment["brew_jobs"])}).'
    )
