"""Prepare a small combined experiment: 2x2 factorial cells + same-source clones.

Budget-capped sibling of C2b / C2c. Bundles two mini-studies in one spec:

  2x2 factorial half (C2b-flavor):
      Single source stratum (medium) x 4 cells (easy_close, easy_far, hard_close,
      hard_far) x N reps. Cells cross self-difficulty with target-target distance,
      so adjacent bars isolate one axis at a time. Fixes the C2 motif confound where
      alignment, difficulty, and target-target distance were bundled together.

  same-source clones half (C2c-flavor):
      2 source classes x 1 clone condition x N reps. Both attackers share (source,
      target); A uses repeat slots 0..N-1, B uses N..2N-1, giving independent
      target images and disjoint poison seeds while still attacking the same
      (source, target) pair. Measures the cooperation ceiling.

Defaults: N=6. Total ~36 duals + ~72 brews + ~72 solos ~= 180 new jobs before
solo reuse from prior experiments.
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
    _c2_target_pair_candidates,
    _class_index,
    _pair_overlap_seed,
    _ranked_source_partners,
    _source_target_ranks,
)


CELL_ORDER = ('easy_close', 'easy_far', 'hard_close', 'hard_far')

CELL_CORNER_TARGETS = dict(
    easy_close=(0.0, 0.0),
    easy_far=(0.0, 1.0),
    hard_close=(1.0, 0.0),
    hard_far=(1.0, 1.0),
)

# Single source stratum for the factorial half. "medium" = middle-ranked partner source.
FACTORIAL_STRATUM = 'medium'

# Same-source half: three (source -> airplane) clones spanning source-target distance,
# so the cooperation ceiling can be read off against attack difficulty.
#   near   : bird    (d=0.493)
#   medium : deer    (d=0.605)
#   far    : frog    (d=0.681)
CLONE_PAIRS = (
    ('bird', 'airplane'),
    ('deer', 'airplane'),
    ('frog', 'airplane'),
)


def _medium_source_pair(fixed_source_class_idx, class_names, distance_matrix):
    ranked_partners = _ranked_source_partners(fixed_source_class_idx, class_names, distance_matrix)
    if len(ranked_partners) < 3:
        raise ValueError('Mini planning expects at least 3 eligible partner-source classes.')
    partner = ranked_partners[len(ranked_partners) // 2]
    return dict(
        left_class=fixed_source_class_idx,
        left_class_name=class_names[fixed_source_class_idx],
        right_class=partner['source_class'],
        right_class_name=partner['source_class_name'],
        source_source_distance=partner['source_source_distance'],
        source_source_rank=partner['source_source_rank'],
        source_stratum=FACTORIAL_STRATUM,
        source_pair_label=f'{class_names[fixed_source_class_idx]}_vs_{partner["source_class_name"]}',
    )


def _corner_score(candidate, difficulty_target, tt_target, difficulty_range, tt_range):
    difficulty = candidate['a_self'] + candidate['b_self']
    tt = candidate['target_target_distance']
    diff_norm = (difficulty - difficulty_range[0]) / max(difficulty_range[1] - difficulty_range[0], 1e-9)
    tt_norm = (tt - tt_range[0]) / max(tt_range[1] - tt_range[0], 1e-9)
    return (diff_norm - difficulty_target) ** 2 + (tt_norm - tt_target) ** 2


def _select_cell_targets(source_pair, class_names, distance_matrix):
    candidates = _c2_target_pair_candidates(
        int(source_pair['left_class']),
        int(source_pair['right_class']),
        class_names,
        distance_matrix,
    )
    difficulties = [candidate['a_self'] + candidate['b_self'] for candidate in candidates]
    tts = [candidate['target_target_distance'] for candidate in candidates]
    difficulty_range = (min(difficulties), max(difficulties))
    tt_range = (min(tts), max(tts))

    used_target_pairs = set()
    selections = {}
    for cell in CELL_ORDER:
        difficulty_target, tt_target = CELL_CORNER_TARGETS[cell]
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                _corner_score(candidate, difficulty_target, tt_target, difficulty_range, tt_range),
                candidate['target_a_class'],
                candidate['target_b_class'],
            ),
        )
        chosen = None
        for cell_rank, candidate in enumerate(ranked, start=1):
            target_pair = (candidate['target_a_class'], candidate['target_b_class'])
            if target_pair in used_target_pairs:
                continue
            chosen = dict(candidate, cell=cell, cell_rank=cell_rank)
            break
        if chosen is None:
            raise ValueError(
                f'Unable to assign a distinct target pair for cell {cell} '
                f'for source pair {source_pair["source_pair_label"]}.'
            )
        used_target_pairs.add((chosen['target_a_class'], chosen['target_b_class']))
        selections[cell] = chosen
    return selections


def _ensure_target_ranks(target_class_idx, rankings, class_names, cache):
    if target_class_idx not in cache:
        cache[target_class_idx] = _source_target_ranks(class_names[target_class_idx], rankings)
    return cache[target_class_idx]


def _attacker_request(
    *,
    source_class_idx,
    target_class_idx,
    repeat_slot,
    distance_matrix,
    rankings,
    class_names,
    target_ranks_cache,
):
    ranks = _ensure_target_ranks(target_class_idx, rankings, class_names, target_ranks_cache)
    return dict(
        source_class_idx=source_class_idx,
        target_class_idx=target_class_idx,
        repeat_slot=repeat_slot,
        source_target_distance=float(distance_matrix[target_class_idx, source_class_idx]),
        source_target_rank=ranks[source_class_idx],
    )


def _build_factorial_requests(
    *,
    preparation_context,
    fixed_source_idx,
    target_ranks_cache,
):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']

    source_pair = _medium_source_pair(fixed_source_idx, class_names, distance_matrix)
    cell_specs = _select_cell_targets(source_pair, class_names, distance_matrix)

    dual_requests = []
    for cell in CELL_ORDER:
        spec = cell_specs[cell]
        condition = f'{source_pair["source_pair_label"]}_{cell}'
        for repeat_slot in range(repeats):
            attacker_requests = [
                _attacker_request(
                    source_class_idx=int(source_pair['left_class']),
                    target_class_idx=int(spec['target_a_class']),
                    repeat_slot=repeat_slot,
                    distance_matrix=distance_matrix,
                    rankings=preparation_context.rankings,
                    class_names=class_names,
                    target_ranks_cache=target_ranks_cache,
                ),
                _attacker_request(
                    source_class_idx=int(source_pair['right_class']),
                    target_class_idx=int(spec['target_b_class']),
                    repeat_slot=repeat_slot,
                    distance_matrix=distance_matrix,
                    rankings=preparation_context.rankings,
                    class_names=class_names,
                    target_ranks_cache=target_ranks_cache,
                ),
            ]
            pairing_id = f'{condition}_rep{repeat_slot + 1}'
            dual_requests.append(dict(
                pairing_id=pairing_id,
                attacker_requests=attacker_requests,
                overlap_seed=_pair_overlap_seed(pairing_id, overlap_seed_base),
                metadata=dict(
                    condition=condition,
                    distance_bucket=source_pair['source_stratum'],
                    source_pair_label=source_pair['source_pair_label'],
                    source_stratum=source_pair['source_stratum'],
                    source_source_rank=source_pair['source_source_rank'],
                    half='factorial_2x2',
                    cell=cell,
                    cell_rank=spec['cell_rank'],
                    difficulty_level='easy' if cell.startswith('easy_') else 'hard',
                    target_distance_level='close' if cell.endswith('_close') else 'far',
                    alignment_type=spec['alignment_type'],
                    target_a_class=int(spec['target_a_class']),
                    target_a_class_name=spec['target_a_class_name'],
                    target_b_class=int(spec['target_b_class']),
                    target_b_class_name=spec['target_b_class_name'],
                    a_self=float(spec['a_self']),
                    b_self=float(spec['b_self']),
                    a_cross=float(spec['a_cross']),
                    b_cross=float(spec['b_cross']),
                    source_source_distance=float(source_pair['source_source_distance']),
                    target_target_distance=float(spec['target_target_distance']),
                    cross_alignment_gap=float(spec['cross_alignment_gap']),
                ),
            ))

    factorial_summary = dict(
        source_stratum=source_pair['source_stratum'],
        source_pair_label=source_pair['source_pair_label'],
        source_source_distance=float(source_pair['source_source_distance']),
        cells={
            cell: dict(
                target_a_class_name=cell_specs[cell]['target_a_class_name'],
                target_b_class_name=cell_specs[cell]['target_b_class_name'],
                a_self=float(cell_specs[cell]['a_self']),
                b_self=float(cell_specs[cell]['b_self']),
                target_target_distance=float(cell_specs[cell]['target_target_distance']),
                alignment_type=cell_specs[cell]['alignment_type'],
            )
            for cell in CELL_ORDER
        },
    )
    return dual_requests, factorial_summary


def _build_clone_requests(
    *,
    preparation_context,
    clone_pairs,
    target_ranks_cache,
):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']

    dual_requests = []
    selections_summary = []

    for source_class_name, target_class_name in clone_pairs:
        source_class_idx = _class_index(class_names, source_class_name)
        clone_target_idx = _class_index(class_names, target_class_name)
        if clone_target_idx == source_class_idx:
            raise ValueError(f'Clone target {target_class_name} must differ from source {source_class_name}.')

        for repeat_slot in range(repeats):
            a_slot = repeat_slot
            b_slot = repeat_slot + repeats
            attacker_requests = [
                _attacker_request(
                    source_class_idx=source_class_idx,
                    target_class_idx=clone_target_idx,
                    repeat_slot=slot,
                    distance_matrix=distance_matrix,
                    rankings=preparation_context.rankings,
                    class_names=class_names,
                    target_ranks_cache=target_ranks_cache,
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
                    half='same_source_clone',
                    source_class=source_class_idx,
                    source_class_name=source_class_name,
                    condition_kind='clone',
                    target_a_class=int(clone_target_idx),
                    target_a_class_name=class_names[clone_target_idx],
                    target_b_class=int(clone_target_idx),
                    target_b_class_name=class_names[clone_target_idx],
                    a_self=float(distance_matrix[clone_target_idx, source_class_idx]),
                    b_self=float(distance_matrix[clone_target_idx, source_class_idx]),
                    source_source_distance=0.0,
                    target_target_distance=0.0,
                    a_repeat_slot=a_slot,
                    b_repeat_slot=b_slot,
                ),
            ))

        selections_summary.append(dict(
            source_class=source_class_idx,
            source_class_name=source_class_name,
            clone_target_class=int(clone_target_idx),
            clone_target_class_name=class_names[clone_target_idx],
            source_target_distance=float(distance_matrix[clone_target_idx, source_class_idx]),
        ))

    return dual_requests, selections_summary


def build_c2_mini_plan(
    *,
    preparation_context,
    fixed_source_class_name,
    clone_pairs,
):
    class_names = preparation_context.class_names
    fixed_source_idx = _class_index(class_names, fixed_source_class_name)

    target_ranks_cache = {}
    factorial_requests, factorial_summary = _build_factorial_requests(
        preparation_context=preparation_context,
        fixed_source_idx=fixed_source_idx,
        target_ranks_cache=target_ranks_cache,
    )
    clone_requests, clone_summary = _build_clone_requests(
        preparation_context=preparation_context,
        clone_pairs=clone_pairs,
        target_ranks_cache=target_ranks_cache,
    )

    return build_experiment_plan(
        family='C2mini',
        experiment_metadata=dict(
            repeats=preparation_context.repeats,
            fixed_attacker_a_source_class=fixed_source_idx,
            factorial_cells=list(CELL_ORDER),
            factorial_stratum=FACTORIAL_STRATUM,
            factorial_selection=factorial_summary,
            clone_pairs=[list(pair) for pair in clone_pairs],
            clone_selections=clone_summary,
            victim_seeds=list(preparation_context.planner_defaults['victim_seeds']),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='mini_factorial_2x2_plus_samesource_clones',
        ),
        dual_requests=factorial_requests + clone_requests,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_prepare_arguments(
        parser,
        fixed_attacker_help='Source class held fixed for attacker A in the factorial half.',
    )
    parser.set_defaults(repeats=6)
    args = parser.parse_args()
    preparation_context = load_preparation_context(args)
    experiment_plan = build_c2_mini_plan(
        preparation_context=preparation_context,
        fixed_source_class_name=preparation_context.fixed_attacker_a_source_class,
        clone_pairs=CLONE_PAIRS,
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
        f'Saved C2-mini experiment spec to {preparation_context.experiment_path} '
        f'(duals={len(experiment["dual_jobs"])}, solos={len(experiment["solo_jobs"])}, '
        f'brews={len(experiment["brew_jobs"])}).'
    )
