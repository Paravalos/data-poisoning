"""Experiment planners for dual-attacker studies."""

from __future__ import annotations

import hashlib
import os
import random

from .experiment import SCHEMA_VERSION


C2_MOTIF_ORDER = (
    'own_aligned_easy',
    'own_aligned_far_apart',
    'cross_aligned_swapped',
    'both_hard_far',
)


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
    if len(ranked) == 0:
        raise ValueError(f'No source-target rankings found for target class {target_class_name}.')
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


def _stable_int_seed(value):
    return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)


def _attacker_id(source_class_idx, selection_key, target_class_idx):
    return f'src{source_class_idx}_sel{selection_key}_to{target_class_idx}'


def _ranked_source_partners(fixed_source_class_idx, class_names, distance_matrix):
    ranked = [
        dict(
            source_class=class_idx,
            source_class_name=class_names[class_idx],
            source_source_distance=float(distance_matrix[fixed_source_class_idx, class_idx]),
        )
        for class_idx in range(len(class_names))
        if class_idx != fixed_source_class_idx
    ]
    ranked.sort(key=lambda entry: (entry['source_source_distance'], entry['source_class']))
    for rank, entry in enumerate(ranked, start=1):
        entry['source_source_rank'] = rank
    return ranked


def _source_strata_pairs(fixed_source_class_idx, class_names, distance_matrix):
    ranked_partners = _ranked_source_partners(fixed_source_class_idx, class_names, distance_matrix)
    if len(ranked_partners) < 3:
        raise ValueError('C2 planning expects at least 3 eligible partner-source classes.')

    stratum_positions = dict(
        near=0,
        medium=len(ranked_partners) // 2,
        far=len(ranked_partners) - 1,
    )

    pairs = []
    for stratum, partner_idx in stratum_positions.items():
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


def _alignment_type(a_self, b_self, a_cross, b_cross):
    if a_self < a_cross and b_self < b_cross:
        return 'own_aligned'
    if a_self > a_cross and b_self > b_cross:
        return 'cross_aligned'
    return 'mixed'


def _c2_target_pair_candidates(source_a_class_idx, source_b_class_idx, class_names, distance_matrix):
    excluded_classes = {source_a_class_idx, source_b_class_idx}
    candidates = []
    for target_a_class_idx in range(len(class_names)):
        if target_a_class_idx in excluded_classes:
            continue
        for target_b_class_idx in range(len(class_names)):
            if target_b_class_idx in excluded_classes or target_b_class_idx == target_a_class_idx:
                continue

            a_self = float(distance_matrix[source_a_class_idx, target_a_class_idx])
            b_self = float(distance_matrix[source_b_class_idx, target_b_class_idx])
            a_cross = float(distance_matrix[source_a_class_idx, target_b_class_idx])
            b_cross = float(distance_matrix[source_b_class_idx, target_a_class_idx])
            candidates.append(dict(
                target_a_class=target_a_class_idx,
                target_a_class_name=class_names[target_a_class_idx],
                target_b_class=target_b_class_idx,
                target_b_class_name=class_names[target_b_class_idx],
                a_self=a_self,
                b_self=b_self,
                a_cross=a_cross,
                b_cross=b_cross,
                source_source_distance=float(distance_matrix[source_a_class_idx, source_b_class_idx]),
                target_target_distance=float(distance_matrix[target_a_class_idx, target_b_class_idx]),
                cross_alignment_gap=(a_cross + b_cross) - (a_self + b_self),
                alignment_type=_alignment_type(a_self, b_self, a_cross, b_cross),
            ))
    return candidates


def _c2_motif_sort_key(candidate, motif_label):
    lexical_key = (candidate['target_a_class'], candidate['target_b_class'])
    if motif_label == 'own_aligned_easy':
        return (
            candidate['a_self'] + candidate['b_self'],
            -candidate['cross_alignment_gap'],
            candidate['target_target_distance'],
            lexical_key,
        )
    if motif_label == 'own_aligned_far_apart':
        return (
            -candidate['target_target_distance'],
            candidate['a_self'] + candidate['b_self'],
            -candidate['cross_alignment_gap'],
            lexical_key,
        )
    if motif_label == 'cross_aligned_swapped':
        return (
            candidate['cross_alignment_gap'],
            candidate['a_cross'] + candidate['b_cross'],
            candidate['target_target_distance'],
            lexical_key,
        )
    if motif_label == 'both_hard_far':
        return (
            -(candidate['a_self'] + candidate['b_self']),
            -candidate['target_target_distance'],
            abs(candidate['cross_alignment_gap']),
            lexical_key,
        )
    raise ValueError(f'Unsupported C2 motif {motif_label}.')


def _rank_c2_motif_candidates(source_pair, class_names, distance_matrix):
    candidates = _c2_target_pair_candidates(
        int(source_pair['left_class']),
        int(source_pair['right_class']),
        class_names,
        distance_matrix,
    )
    motif_predicates = {
        'own_aligned_easy': lambda candidate: candidate['alignment_type'] == 'own_aligned',
        'own_aligned_far_apart': lambda candidate: candidate['alignment_type'] == 'own_aligned',
        'cross_aligned_swapped': lambda candidate: candidate['alignment_type'] == 'cross_aligned',
        'both_hard_far': lambda candidate: True,
    }

    ranked = {}
    for motif_label in C2_MOTIF_ORDER:
        motif_candidates = [candidate for candidate in candidates if motif_predicates[motif_label](candidate)]
        if not motif_candidates:
            raise ValueError(
                f'No feasible candidates found for motif {motif_label} '
                f'for source pair {source_pair["source_pair_label"]}.'
            )
        ranked[motif_label] = sorted(
            motif_candidates,
            key=lambda candidate: _c2_motif_sort_key(candidate, motif_label),
        )
    return ranked


def _select_c2_motif_targets(source_pair, class_names, distance_matrix):
    ranked = _rank_c2_motif_candidates(source_pair, class_names, distance_matrix)
    used_target_pairs = set()
    selections = {}

    for motif_label in C2_MOTIF_ORDER:
        chosen = None
        for motif_rank, candidate in enumerate(ranked[motif_label], start=1):
            target_pair = (candidate['target_a_class'], candidate['target_b_class'])
            if target_pair in used_target_pairs:
                continue
            chosen = dict(candidate, motif_label=motif_label, motif_rank=motif_rank)
            break
        if chosen is None:
            raise ValueError(
                f'Unable to assign a distinct target pair for motif {motif_label} '
                f'for source pair {source_pair["source_pair_label"]}.'
            )
        used_target_pairs.add((chosen['target_a_class'], chosen['target_b_class']))
        selections[motif_label] = chosen

    return selections


def _repeat_poison_seed(source_class_idx, repeat_slot):
    return f'src{source_class_idx}_rep{repeat_slot}'


def _pair_overlap_seed(pairing_id, overlap_seed_base=0):
    return overlap_seed_base + _stable_int_seed(f'overlap:{pairing_id}')


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


def build_c2_experiment(
    *,
    experiment_id,
    class_names,
    rankings,
    distance_matrix,
    class_to_valid_indices,
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
    """Build a concrete C2 experiment JSON payload."""
    fixed_attacker_a_source_class_idx = _class_index(class_names, fixed_attacker_a_source_class)
    source_pairs = _source_strata_pairs(fixed_attacker_a_source_class_idx, class_names, distance_matrix)
    source_classes = {int(pair['left_class']) for pair in source_pairs} | {int(pair['right_class']) for pair in source_pairs}
    sampled_target_indices = _sample_target_indices(class_to_valid_indices, source_classes, repeats, planning_seed)

    experiment_root = output_root
    brew_dir = f'{experiment_root}/brews'
    solo_dir = f'{experiment_root}/solo'
    dual_dir = f'{experiment_root}/dual'

    brew_jobs, solo_jobs, dual_jobs = [], [], []
    artifact_by_attacker = {}
    attacker_specs = {}
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
                attackers = []
                for source_class_idx, target_class_idx in (
                    (int(source_pair['left_class']), int(motif_spec['target_a_class'])),
                    (int(source_pair['right_class']), int(motif_spec['target_b_class'])),
                ):
                    target_index = int(sampled_target_indices[source_class_idx][repeat_slot])
                    attacker_key = (source_class_idx, repeat_slot, target_class_idx)
                    if attacker_key not in attacker_specs:
                        if target_class_idx not in source_target_ranks_by_target:
                            source_target_ranks_by_target[target_class_idx] = _source_target_ranks(
                                class_names[target_class_idx],
                                rankings,
                            )
                        attacker_id = _attacker_id(source_class_idx, repeat_slot, target_class_idx)
                        brew_job_id = f'brew_{attacker_id}'
                        artifact_path = os.path.join(brew_dir, f'{brew_job_id}.pt')
                        poison_ids_seed = _repeat_poison_seed(source_class_idx, repeat_slot)
                        arg_overrides = dict(
                            poisonkey=f'{source_class_idx}-{target_class_idx}-{target_index}',
                            poison_ids_seed=poison_ids_seed,
                            name=brew_job_id,
                            targets=1,
                        )
                        attacker_meta = dict(
                            attacker_id=attacker_id,
                            poisonkey=arg_overrides['poisonkey'],
                            poison_ids_seed=poison_ids_seed,
                            brew_job_id=brew_job_id,
                            selection_key=repeat_slot,
                            repeat_slot=repeat_slot,
                            target_index=target_index,
                            source_class=source_class_idx,
                            target_true_class=source_class_idx,
                            target_adv_class=target_class_idx,
                            source_target_distance=float(distance_matrix[target_class_idx, source_class_idx]),
                            source_target_rank=source_target_ranks_by_target[target_class_idx][source_class_idx],
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

                pairing_id = f'{condition}_rep{repeat_slot + 1}'
                dual_jobs.append(dict(
                    job_id=f'dual_{pairing_id}',
                    pairing_id=pairing_id,
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
                    attackers=attackers,
                    brew_artifact_paths=[artifact_by_attacker[attackers[0]['attacker_id']],
                                         artifact_by_attacker[attackers[1]['attacker_id']]],
                    victim_seeds=list(victim_seeds),
                    overlap_seed=_pair_overlap_seed(pairing_id, overlap_seed_base),
                    overlap_policy='assign_one_owner',
                    source_source_distance=float(source_pair['source_source_distance']),
                    target_target_distance=float(motif_spec['target_target_distance']),
                    cross_alignment_gap=float(motif_spec['cross_alignment_gap']),
                    arg_overrides=dict(name=f'dual_{pairing_id}', poisonkey=None),
                    output_path=os.path.join(dual_dir, f'dual_{pairing_id}.csv'),
                ))

    return dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        family='C2',
        class_names=list(class_names),
        metadata=dict(
            fixed_attacker_a_source_class=fixed_attacker_a_source_class_idx,
            planning_seed=planning_seed,
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
            distance_artifact_path=distance_artifact_path,
            sampled_target_indices={str(class_idx): indices for class_idx, indices in sampled_target_indices.items()},
            pair_selection_strategy='fixed_attacker_a_source_strata_target_sweep',
        ),
        scheduler=scheduler or {},
        common_args=common_args,
        output_root=experiment_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )
