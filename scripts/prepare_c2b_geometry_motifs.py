"""Prepare a C2b 2x2 factorial-motif experiment.

C2 bundled self-difficulty, target-target distance, and alignment into four motifs that
are hard to disentangle visually. C2b keeps the five source-distance strata of C2 but
replaces the motifs with a clean 2x2 factorial of:

    self-difficulty  ({easy, hard})  X  target-target distance  ({close, far})

yielding four cells per source pair: easy_close, easy_far, hard_close, hard_far. Each
cell picks the single target pair whose geometry best hits that corner, so adjacent
bars in a plot answer "does axis X matter?" with difficulty or target-target held
constant. No regression required.
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


CELL_ORDER = (
    'easy_close',
    'easy_far',
    'hard_close',
    'hard_far',
)

STRATUM_ORDER = ('near', 'mid_near', 'medium', 'mid_far', 'far')


def _five_source_strata_pairs(fixed_source_class_idx, class_names, distance_matrix):
    """Five evenly-spaced source strata over ranked partners (matches original C2 layout)."""
    ranked_partners = _ranked_source_partners(fixed_source_class_idx, class_names, distance_matrix)
    if len(ranked_partners) < 5:
        raise ValueError('C2b planning expects at least 5 eligible partner-source classes.')

    n = len(ranked_partners)
    stratum_positions = dict(
        near=0,
        mid_near=n // 4,
        medium=n // 2,
        mid_far=(3 * n) // 4,
        far=n - 1,
    )

    seen_indices = set()
    pairs = []
    for stratum in STRATUM_ORDER:
        partner_idx = stratum_positions[stratum]
        if partner_idx in seen_indices:
            raise ValueError(f'C2b source strata collided at position {partner_idx} for stratum {stratum}.')
        seen_indices.add(partner_idx)
        partner = ranked_partners[partner_idx]
        pairs.append(dict(
            left_class=fixed_source_class_idx,
            left_class_name=class_names[fixed_source_class_idx],
            right_class=partner['source_class'],
            right_class_name=partner['source_class_name'],
            source_source_distance=partner['source_source_distance'],
            source_source_rank=partner['source_source_rank'],
            source_stratum=stratum,
            source_pair_label=f'{class_names[fixed_source_class_idx]}_vs_{partner["source_class_name"]}',
        ))
    return pairs


def _corner_score(candidate, difficulty_target, tt_target, difficulty_range, tt_range):
    """Normalized distance from the candidate to the corner-of-interest in (difficulty, d_TT) space.

    Balanced across the two axes so neither dominates when one has a tighter spread.
    """
    difficulty = candidate['a_self'] + candidate['b_self']
    tt = candidate['target_target_distance']
    diff_norm = (difficulty - difficulty_range[0]) / max(difficulty_range[1] - difficulty_range[0], 1e-9)
    tt_norm = (tt - tt_range[0]) / max(tt_range[1] - tt_range[0], 1e-9)
    return (diff_norm - difficulty_target) ** 2 + (tt_norm - tt_target) ** 2


CELL_CORNER_TARGETS = dict(
    easy_close=(0.0, 0.0),
    easy_far=(0.0, 1.0),
    hard_close=(1.0, 0.0),
    hard_far=(1.0, 1.0),
)


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


def build_c2b_plan(*, preparation_context):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']
    fixed_attacker_a_source_class_idx = _class_index(
        class_names,
        preparation_context.fixed_attacker_a_source_class,
    )
    source_pairs = _five_source_strata_pairs(
        fixed_attacker_a_source_class_idx, class_names, distance_matrix,
    )

    cell_specs_by_pair = {
        source_pair['source_pair_label']: _select_cell_targets(
            source_pair, class_names, distance_matrix,
        )
        for source_pair in source_pairs
    }
    source_target_ranks_by_target = {}
    dual_requests = []

    for source_pair in source_pairs:
        cell_specs = cell_specs_by_pair[source_pair['source_pair_label']]
        for cell in CELL_ORDER:
            spec = cell_specs[cell]
            condition = f'{source_pair["source_pair_label"]}_{cell}'
            for repeat_slot in range(repeats):
                attacker_requests = []
                for source_class_idx, target_class_idx in (
                    (int(source_pair['left_class']), int(spec['target_a_class'])),
                    (int(source_pair['right_class']), int(spec['target_b_class'])),
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
                        source_pair_label=source_pair['source_pair_label'],
                        source_stratum=source_pair['source_stratum'],
                        source_source_rank=source_pair['source_source_rank'],
                        cell=cell,
                        cell_rank=spec['cell_rank'],
                        difficulty_level='easy' if cell.startswith('easy_') else 'hard',
                        target_distance_level='close' if cell.endswith('_close') else 'far',
                        alignment_type=spec['alignment_type'],
                        target_a_class=int(spec['target_a_class']),
                        target_a_class_name=spec['target_a_class_name'],
                        target_b_class=int(spec['target_b_class']),
                        target_b_class_name=spec['target_b_class_name'],
                        S=float(source_pair['source_source_distance']),
                        T=float(spec['target_target_distance']),
                        G=float(spec['cross_alignment_gap']),
                        a_self=float(spec['a_self']),
                        b_self=float(spec['b_self']),
                        a_cross=float(spec['a_cross']),
                        b_cross=float(spec['b_cross']),
                        source_source_distance=float(source_pair['source_source_distance']),
                        target_target_distance=float(spec['target_target_distance']),
                        cross_alignment_gap=float(spec['cross_alignment_gap']),
                    ),
                ))

    return build_experiment_plan(
        family='C2b',
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
            cell_labels=list(CELL_ORDER),
            victim_seeds=list(preparation_context.planner_defaults['victim_seeds']),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='fixed_attacker_a_source_strata_2x2_geometry_cells',
        ),
        dual_requests=dual_requests,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_prepare_arguments(
        parser,
        fixed_attacker_help='Source class held fixed for attacker A while C2b sweeps target geometry.',
    )
    parser.set_defaults(repeats=8)
    args = parser.parse_args()
    preparation_context = load_preparation_context(args)
    experiment_plan = build_c2b_plan(preparation_context=preparation_context)
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
        f'Saved C2b experiment spec to {preparation_context.experiment_path} '
        f'(duals={len(experiment["dual_jobs"])}, solos={len(experiment["solo_jobs"])}, '
        f'brews={len(experiment["brew_jobs"])}).'
    )
