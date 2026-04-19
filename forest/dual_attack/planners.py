"""Experiment planners for dual-attacker studies."""

from __future__ import annotations

import hashlib
import math
import os

from forest.data.kettle_det_experiment import select_deterministic_poison_ids

from .experiment import SCHEMA_VERSION, build_args_namespace


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

def _stable_int_seed(value):
    return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)


def _attacker_id(source_class_idx, repeat_slot, target_class_idx, target_slot=0):
    slot_suffix = '' if int(target_slot) == 0 else f'x{int(target_slot)}'
    return f'src{source_class_idx}_sel{repeat_slot}{slot_suffix}_to{target_class_idx}'


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


def _build_attacker_spec(
    *,
    source_class_idx,
    target_class_idx,
    target_index,
    repeat_slot,
    source_target_distance,
    source_target_rank,
    brew_dir,
    target_slot=0,
):
    attacker_id = _attacker_id(source_class_idx, repeat_slot, target_class_idx, target_slot=target_slot)
    brew_job_id = f'brew_{attacker_id}'
    poison_ids_seed = _repeat_poison_seed(source_class_idx, repeat_slot)
    return dict(
        job_id=brew_job_id,
        attacker=dict(
            attacker_id=attacker_id,
            poisonkey=f'{source_class_idx}-{target_class_idx}-{target_index}',
            poison_ids_seed=poison_ids_seed,
            brew_job_id=brew_job_id,
            repeat_slot=repeat_slot,
            target_index=target_index,
            source_class=source_class_idx,
            target_true_class=source_class_idx,
            target_adv_class=target_class_idx,
            source_target_distance=float(source_target_distance),
            source_target_rank=source_target_rank,
        ),
        arg_overrides=dict(
            poisonkey=f'{source_class_idx}-{target_class_idx}-{target_index}',
            poison_ids_seed=poison_ids_seed,
            name=brew_job_id,
            targets=1,
        ),
        artifact_path=os.path.join(brew_dir, f'{brew_job_id}.pt'),
    )


def _pair_overlap_seed(pairing_id, overlap_seed_base=0):
    return overlap_seed_base + _stable_int_seed(f'overlap:{pairing_id}')


def _poison_budget_count(class_to_train_indices, budget):
    train_size = sum(len(indices) for indices in class_to_train_indices.values())
    if train_size <= 0:
        raise ValueError('Expected a non-empty training index mapping.')
    return int(math.ceil(float(budget) * train_size))


def _planned_attacker_poison_ids(attacker, class_to_train_indices, budget, randomize_poison_ids):
    explicit_poison_ids = attacker.get('explicit_poison_ids')
    if explicit_poison_ids is not None:
        return [int(index) for index in explicit_poison_ids]

    poison_class = int(attacker['target_adv_class'])
    class_ids = list(class_to_train_indices[poison_class])
    poison_num = min(_poison_budget_count(class_to_train_indices, budget), len(class_ids))
    return [int(index) for index in select_deterministic_poison_ids(
        class_ids,
        poison_num,
        attacker['poisonkey'],
        randomize_poison_ids=randomize_poison_ids,
        poison_ids_seed=attacker.get('poison_ids_seed'),
    )]


def _normalize_overlap_percentages(overlap_percentages):
    normalized = []
    for raw_value in overlap_percentages:
        value = int(raw_value)
        if value < 0 or value > 100:
            raise ValueError(f'Overlap percentage must be between 0 and 100, found {value}.')
        if value not in normalized:
            normalized.append(value)
    if len(normalized) == 0:
        raise ValueError('Expected at least one overlap percentage.')
    return normalized


def _build_explicit_overlap_sets(left_base_ids, right_base_ids, class_ids, target_shared_count):
    left = {int(index) for index in left_base_ids}
    right = {int(index) for index in right_base_ids}
    if len(left) != len(left_base_ids) or len(right) != len(right_base_ids):
        raise ValueError('C5 planning expects unique poison ids per attacker.')
    if len(left) != len(right):
        raise ValueError('C5 planning expects both attackers to use the same poison budget.')
    if target_shared_count < 0 or target_shared_count > len(left):
        raise ValueError(f'Invalid shared-id target {target_shared_count} for poison budget {len(left)}.')

    all_class_ids = [int(index) for index in class_ids]
    fresh_ids = [index for index in all_class_ids if index not in left and index not in right]

    reduce_on_left = False
    while len(left & right) > target_shared_count:
        shared_id = min(left & right)
        if len(fresh_ids) == 0:
            raise ValueError('Unable to reduce overlap: ran out of fresh poison ids.')
        replacement = fresh_ids.pop(0)
        if reduce_on_left:
            left.remove(shared_id)
            left.add(replacement)
        else:
            right.remove(shared_id)
            right.add(replacement)
        reduce_on_left = not reduce_on_left

    add_to_left = False
    while len(left & right) < target_shared_count:
        left_only = sorted(left - right)
        right_only = sorted(right - left)
        if len(left_only) == 0 or len(right_only) == 0:
            raise ValueError('Unable to increase overlap further with the available base poison ids.')

        if add_to_left:
            shared_id = right_only[0]
            dropped_id = left_only[-1]
            left.remove(dropped_id)
            left.add(shared_id)
        else:
            shared_id = left_only[0]
            dropped_id = right_only[-1]
            right.remove(dropped_id)
            right.add(shared_id)
        add_to_left = not add_to_left

    return sorted(left), sorted(right)


def _c1_condition_jobs(c1_experiment, pair_condition):
    matching_jobs = [
        job for job in c1_experiment.get('dual_jobs', [])
        if job.get('condition') == pair_condition
    ]
    if len(matching_jobs) == 0:
        raise ValueError(
            f'Could not find any C1 dual jobs for condition {pair_condition} '
            f'in experiment {c1_experiment.get("experiment_id", "<unknown>")}.'
        )
    return sorted(matching_jobs, key=lambda job: (job.get('pairing_id', ''), job['job_id']))

def build_c5_experiment(
    *,
    experiment_id,
    c1_experiment,
    class_to_train_indices,
    pair_condition,
    overlap_percentages,
    output_root,
    scheduler=None,
    source_experiment_path=None,
    merge_rule='sum_clipped',
):
    """Build a C5 overlap-collision experiment derived from a C1 spec."""
    if c1_experiment.get('family') != 'C1':
        raise ValueError(f'C5 planning expects a C1 source experiment, found {c1_experiment.get("family")}.')
    if merge_rule != 'sum_clipped':
        raise ValueError(f'Unsupported C5 merge rule {merge_rule}.')

    selected_jobs = _c1_condition_jobs(c1_experiment, pair_condition)
    common_args = dict(c1_experiment.get('common_args', {}))
    budget = common_args.get('budget', build_args_namespace(common_args).budget)
    randomize_poison_ids = bool(common_args.get('randomize_deterministic_poison_ids', False))
    class_names = list(c1_experiment['class_names'])
    overlap_percentages = _normalize_overlap_percentages(overlap_percentages)
    poison_budget_count = _poison_budget_count(class_to_train_indices, budget)

    experiment_root = output_root
    brew_dir = f'{experiment_root}/brews'
    solo_dir = f'{experiment_root}/solo'
    dual_dir = f'{experiment_root}/dual'

    brew_jobs, solo_jobs, dual_jobs = [], [], []
    sampled_target_indices = {}

    for repeat_idx, source_job in enumerate(selected_jobs):
        left_base = dict(source_job['attackers'][0])
        right_base = dict(source_job['attackers'][1])
        for attacker in (left_base, right_base):
            sampled_target_indices.setdefault(str(attacker['source_class']), [])
            if attacker['target_index'] not in sampled_target_indices[str(attacker['source_class'])]:
                sampled_target_indices[str(attacker['source_class'])].append(attacker['target_index'])

        left_base_poison_ids = _planned_attacker_poison_ids(
            left_base,
            class_to_train_indices,
            budget,
            randomize_poison_ids,
        )
        right_base_poison_ids = _planned_attacker_poison_ids(
            right_base,
            class_to_train_indices,
            budget,
            randomize_poison_ids,
        )
        poison_class_ids = list(class_to_train_indices[int(left_base['target_adv_class'])])

        for overlap_percentage in overlap_percentages:
            overlap_count = int(round(len(left_base_poison_ids) * overlap_percentage / 100.0))
            left_poison_ids, right_poison_ids = _build_explicit_overlap_sets(
                left_base_poison_ids,
                right_base_poison_ids,
                poison_class_ids,
                overlap_count,
            )
            attackers = []
            for base_attacker, explicit_poison_ids in (
                (left_base, left_poison_ids),
                (right_base, right_poison_ids),
            ):
                overlap_suffix = f'ovl{overlap_percentage}'
                attacker_id = f'{base_attacker["attacker_id"]}_{overlap_suffix}'
                brew_job_id = f'brew_{attacker_id}'
                artifact_path = os.path.join(brew_dir, f'{brew_job_id}.pt')
                attacker_meta = dict(base_attacker)
                attacker_meta.pop('poison_ids_seed', None)
                attacker_meta.update(
                    attacker_id=attacker_id,
                    brew_job_id=brew_job_id,
                    explicit_poison_ids=list(explicit_poison_ids),
                    c1_attacker_id=base_attacker['attacker_id'],
                    overlap_percentage=overlap_percentage,
                    overlap_fraction=overlap_percentage / 100.0,
                    overlap_target_count=overlap_count,
                )
                brew_jobs.append(dict(
                    job_id=brew_job_id,
                    attacker=attacker_meta,
                    arg_overrides=dict(
                        poisonkey=base_attacker['poisonkey'],
                        explicit_poison_ids=list(explicit_poison_ids),
                        name=brew_job_id,
                        targets=1,
                    ),
                    artifact_path=artifact_path,
                ))
                solo_jobs.append(dict(
                    job_id=f'solo_{attacker_id}',
                    attacker=attacker_meta,
                    brew_artifact_path=artifact_path,
                    victim_seeds=list(source_job['victim_seeds']),
                    arg_overrides=dict(name=f'solo_{attacker_id}', poisonkey=None),
                    output_path=os.path.join(solo_dir, f'solo_{attacker_id}.csv'),
                ))
                attackers.append(attacker_meta)

            pairing_id = f'{pair_condition}_ovl{overlap_percentage}_rep{repeat_idx + 1}'
            dual_jobs.append(dict(
                job_id=f'dual_{pairing_id}',
                pairing_id=pairing_id,
                c1_pairing_id=source_job.get('pairing_id', ''),
                condition=pair_condition,
                distance_bucket='overlap_sweep',
                source_pair_label=f'{class_names[left_base["source_class"]]}_vs_{class_names[right_base["source_class"]]}',
                symmetric='',
                pair_distance_rank='',
                pair_distance_rank_fraction='',
                attackers=attackers,
                brew_artifact_paths=[
                    os.path.join(brew_dir, f'brew_{attackers[0]["attacker_id"]}.pt'),
                    os.path.join(brew_dir, f'brew_{attackers[1]["attacker_id"]}.pt'),
                ],
                victim_seeds=list(source_job['victim_seeds']),
                overlap_seed=_pair_overlap_seed(pairing_id, 0),
                overlap_policy='explicit_shared_ids',
                merge_rule=merge_rule,
                overlap_percentage=overlap_percentage,
                overlap_fraction=overlap_percentage / 100.0,
                overlap_target_count=overlap_count,
                source_source_distance=source_job.get('source_source_distance', ''),
                arg_overrides=dict(name=f'dual_{pairing_id}', poisonkey=None),
                output_path=os.path.join(dual_dir, f'dual_{pairing_id}.csv'),
            ))

    fixed_source_class = int(selected_jobs[0]['attackers'][0]['source_class'])
    partner_source_class = int(selected_jobs[0]['attackers'][1]['source_class'])
    shared_target_class = int(selected_jobs[0]['attackers'][0]['target_adv_class'])
    repeats = len(selected_jobs)

    return dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        family='C5',
        class_names=class_names,
        metadata=dict(
            shared_target_class=shared_target_class,
            fixed_attacker_a_source_class=fixed_source_class,
            partner_source_class=partner_source_class,
            pair_condition=pair_condition,
            repeats=repeats,
            victim_seeds=list(selected_jobs[0]['victim_seeds']),
            source_experiment_id=c1_experiment['experiment_id'],
            source_experiment_path=source_experiment_path,
            overlap_percentages=list(overlap_percentages),
            merge_rule=merge_rule,
            poison_budget_count=poison_budget_count,
            sampled_target_indices=sampled_target_indices,
            pair_selection_strategy='c1_pair_overlap_collision_sweep',
        ),
        scheduler=scheduler or dict(c1_experiment.get('scheduler', {})),
        common_args=common_args,
        output_root=experiment_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )
