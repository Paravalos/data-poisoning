"""Prepare C6 target-pair similarity experiment.

C6 keeps the attack recipe and training settings from the recent dual-attacker
experiments, but changes target-image selection. For each shared true target
class, target pairs are selected by cosine distance between clean ResNet-18
penultimate image representations:

  closest, median / 50th percentile, 75th percentile

Target images are disjoint across all selected bins within a class. Each fixed
target pair is repeated with independent poison-id seeds, then evaluated as
A alone, B alone, A+B combined, A alone with 2x budget, and B alone with 2x
budget.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

import forest
import forest.utils
from forest.data.datasets import Subset, construct_datasets
from forest.dual_attack.config import common_args_defaults, planner_defaults, scheduler_defaults
from forest.dual_attack.experiment import SCHEMA_VERSION, build_args_namespace, save_experiment
from forest.dual_attack.planners import _class_index, _pair_overlap_seed, _source_target_ranks
from forest.dual_attack.prepare import class_to_dataset_indices, output_paths_from_output_arg
from forest.victims.models import get_model


TARGET_POISON_PAIRS = (
    ('airplane', 'dog'),
    ('bird', 'truck'),
    ('cat', 'airplane'),
)

PAIR_BINS = (
    dict(name='closest', quantile=0.0),
    dict(name='median', quantile=0.50),
    dict(name='q75', quantile=0.75),
)

# Match the recent standard dual-attack setting.
EPS_OVERRIDE = 8

SCHEDULER_PER_STAGE_OVERRIDES = dict(
    brew=dict(time='1:00:00'),
    solo=dict(time='1:30:00'),
    dual=dict(time='1:30:00'),
)


def _feature_extractor(model):
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    headless_model = torch.nn.Sequential(*list(target_model.children())[:-1], torch.nn.Flatten())
    headless_model.to(next(target_model.parameters()).device)
    headless_model.eval()
    return headless_model


def _load_clean_model(common_args, distance_artifact, setup, *, clean_model_override, train_if_missing):
    args = build_args_namespace(common_args)
    model = get_model(args.net[0], args.dataset, pretrained=args.pretrained_model)
    model.to(**setup)

    clean_model_path = clean_model_override or distance_artifact.get('clean_model_cache_path')
    if clean_model_path:
        clean_model_path = os.path.expanduser(clean_model_path)
    if clean_model_path and os.path.isfile(clean_model_path):
        state = torch.load(clean_model_path, map_location=setup['device'])
        model.load_state_dict(state)
        model.eval()
        return model, clean_model_path

    if not train_if_missing:
        raise FileNotFoundError(
            'Clean model cache was not found. Expected '
            f'{clean_model_path or "<missing clean_model_cache_path>"}. '
            'Re-run with --train-clean-if-missing to train it during preparation, '
            'or provide a distance artifact whose clean_model_cache_path exists.'
        )

    victim = forest.Victim(args, setup=setup)
    data = forest.Kettle(
        args,
        victim.defs.batch_size,
        victim.defs.augmentations,
        victim.defs.mixing_method,
        setup=setup,
    )
    victim.train(data, max_epoch=args.max_epoch)
    trained_model = victim.model.module if isinstance(victim.model, torch.nn.DataParallel) else victim.model
    trained_model.eval()
    return trained_model, (
        victim._compute_clean_model_cache_path()
        if hasattr(victim, '_compute_clean_model_cache_path') else ''
    )


def _compute_valid_features(model, validset, valid_indices, batch_size, setup):
    feature_model = _feature_extractor(model)
    subset = Subset(validset, indices=list(valid_indices))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    features = []
    indices = []
    with torch.no_grad():
        for images, _labels, batch_indices in loader:
            images = images.to(**setup)
            features.append(feature_model(images).detach().cpu())
            indices.extend(int(index) for index in batch_indices.tolist())
    if not features:
        raise ValueError('No valid-set features were extracted.')
    return torch.cat(features, dim=0), indices


def _candidate_pairs_for_class(features, indices):
    normalized = F.normalize(features, dim=1)
    distance_matrix = 1 - normalized @ normalized.t()
    left, right = torch.triu_indices(distance_matrix.shape[0], distance_matrix.shape[1], offset=1)
    distances = distance_matrix[left, right]
    pairs = [
        dict(
            left_index=int(indices[int(i)]),
            right_index=int(indices[int(j)]),
            feature_cosine_distance=float(distance),
            feature_cosine_similarity=float(1.0 - distance),
        )
        for i, j, distance in zip(left.tolist(), right.tolist(), distances.tolist())
    ]
    return pairs, distances


def _select_pairs_for_bin(candidates, used_indices, *, bin_name, quantile_value, pairs_per_bin):
    if bin_name == 'closest':
        ranked = sorted(
            candidates,
            key=lambda pair: (
                pair['feature_cosine_distance'],
                pair['left_index'],
                pair['right_index'],
            ),
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda pair: (
                abs(pair['feature_cosine_distance'] - quantile_value),
                pair['feature_cosine_distance'],
                pair['left_index'],
                pair['right_index'],
            ),
        )

    selected = []
    for pair in ranked:
        if pair['left_index'] in used_indices or pair['right_index'] in used_indices:
            continue
        selected.append(dict(pair, pair_bin=bin_name, bin_quantile_value=float(quantile_value)))
        used_indices.add(pair['left_index'])
        used_indices.add(pair['right_index'])
        if len(selected) == pairs_per_bin:
            break

    if len(selected) != pairs_per_bin:
        raise ValueError(
            f'Could only select {len(selected)} disjoint pairs for {bin_name}; '
            f'needed {pairs_per_bin}.'
        )
    return selected


def _gradient_cosine(model, validset, left_index, right_index, adv_class, setup):
    model.eval()
    params = [param for param in model.parameters() if param.requires_grad]

    def target_grad(index):
        image, _label, _idx = validset[int(index)]
        image = image.unsqueeze(0).to(**setup)
        label = torch.tensor([int(adv_class)], device=setup['device'])
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(image), label)
        return torch.autograd.grad(loss, params, retain_graph=False)

    left_grad = target_grad(left_index)
    right_grad = target_grad(right_index)
    dot = torch.zeros((), device=setup['device'])
    left_norm_sq = torch.zeros((), device=setup['device'])
    right_norm_sq = torch.zeros((), device=setup['device'])
    for left_part, right_part in zip(left_grad, right_grad):
        dot += (left_part * right_part).sum()
        left_norm_sq += left_part.pow(2).sum()
        right_norm_sq += right_part.pow(2).sum()
    denom = left_norm_sq.sqrt() * right_norm_sq.sqrt()
    if denom.item() == 0:
        return float('nan')
    return float((dot / denom).detach().cpu().item())


def _target_pair_selections(
    *,
    model,
    validset,
    class_to_valid_indices,
    class_names,
    target_poison_pairs,
    pairs_per_bin,
    batch_size,
    setup,
):
    selections = []
    summaries = []
    for target_name, poison_name in target_poison_pairs:
        target_idx = _class_index(class_names, target_name)
        poison_idx = _class_index(class_names, poison_name)
        features, indices = _compute_valid_features(
            model,
            validset,
            class_to_valid_indices[target_idx],
            batch_size,
            setup,
        )
        candidates, all_distances = _candidate_pairs_for_class(features, indices)
        used_indices = set()
        class_pairs = []
        for bin_spec in PAIR_BINS:
            quantile_value = (
                float(all_distances.min().item())
                if bin_spec['name'] == 'closest'
                else float(torch.quantile(all_distances, bin_spec['quantile']).item())
            )
            class_pairs.extend(_select_pairs_for_bin(
                candidates,
                used_indices,
                bin_name=bin_spec['name'],
                quantile_value=quantile_value,
                pairs_per_bin=pairs_per_bin,
            ))

        for pair_number, pair in enumerate(class_pairs, start=1):
            gradient_cosine = _gradient_cosine(
                model,
                validset,
                pair['left_index'],
                pair['right_index'],
                poison_idx,
                setup,
            )
            selections.append(dict(
                pair,
                pair_number=pair_number,
                target_class=target_idx,
                target_class_name=target_name,
                poison_class=poison_idx,
                poison_class_name=poison_name,
                gradient_cosine=gradient_cosine,
            ))

        summaries.append(dict(
            target_class=target_idx,
            target_class_name=target_name,
            poison_class=poison_idx,
            poison_class_name=poison_name,
            selected_pairs=[
                dict(
                    pair_number=pair['pair_number'],
                    pair_bin=pair['pair_bin'],
                    left_index=pair['left_index'],
                    right_index=pair['right_index'],
                    feature_cosine_distance=pair['feature_cosine_distance'],
                    feature_cosine_similarity=pair['feature_cosine_similarity'],
                    gradient_cosine=pair['gradient_cosine'],
                )
                for pair in selections
                if pair['target_class'] == target_idx and pair['poison_class'] == poison_idx
            ],
        ))
    return selections, summaries


def _format_budget_suffix(multiplier):
    return 'b1x' if multiplier == 1 else f'b{multiplier}x'


def _build_attacker_spec(
    *,
    pair,
    role,
    target_index,
    repeat_slot,
    budget_multiplier,
    base_budget,
    source_target_distance,
    source_target_rank,
    brew_dir,
):
    budget_suffix = _format_budget_suffix(budget_multiplier)
    pair_key = (
        f'{pair["target_class_name"]}_to_{pair["poison_class_name"]}_'
        f'{pair["pair_bin"]}_p{pair["pair_number"]:02d}'
    )
    attacker_id = f'c6_{pair_key}_rep{repeat_slot + 1}_{role}_{budget_suffix}'
    brew_job_id = f'brew_{attacker_id}'
    poison_ids_seed = f'c6_{pair_key}_rep{repeat_slot + 1}_{role}'
    effective_budget = float(base_budget) * float(budget_multiplier)
    arg_overrides = dict(
        poisonkey=f'{pair["target_class"]}-{pair["poison_class"]}-{target_index}',
        poison_ids_seed=poison_ids_seed,
        name=brew_job_id,
        targets=1,
    )
    if budget_multiplier != 1:
        arg_overrides['budget'] = effective_budget

    attacker = dict(
        attacker_id=attacker_id,
        poisonkey=arg_overrides['poisonkey'],
        poison_ids_seed=poison_ids_seed,
        brew_job_id=brew_job_id,
        repeat_slot=repeat_slot,
        target_index=int(target_index),
        source_class=int(pair['target_class']),
        target_true_class=int(pair['target_class']),
        target_adv_class=int(pair['poison_class']),
        source_target_distance=float(source_target_distance),
        source_target_rank=source_target_rank,
        selection_key=int(pair['pair_number']),
        pair_key=pair_key,
        pair_bin=pair['pair_bin'],
        pair_number=int(pair['pair_number']),
        attacker_role=role,
        budget_multiplier=budget_multiplier,
        base_budget=float(base_budget),
        effective_budget=effective_budget,
        feature_cosine_distance=float(pair['feature_cosine_distance']),
        feature_cosine_similarity=float(pair['feature_cosine_similarity']),
        gradient_cosine=float(pair['gradient_cosine']),
    )
    return dict(
        job_id=brew_job_id,
        attacker=attacker,
        arg_overrides=arg_overrides,
        artifact_path=os.path.join(brew_dir, f'{brew_job_id}.pt'),
    )


def _solo_job(attacker_spec, solo_dir, victim_seeds):
    attacker = attacker_spec['attacker']
    arg_overrides = dict(name=f'solo_{attacker["attacker_id"]}', poisonkey=None)
    brew_arg_overrides = {}
    if attacker['budget_multiplier'] != 1:
        brew_arg_overrides['budget'] = attacker['effective_budget']
        arg_overrides['budget'] = attacker['effective_budget']
    job = dict(
        job_id=f'solo_{attacker["attacker_id"]}',
        attacker=attacker,
        brew_artifact_path=attacker_spec['artifact_path'],
        victim_seeds=list(victim_seeds),
        arg_overrides=arg_overrides,
        output_path=os.path.join(solo_dir, f'solo_{attacker["attacker_id"]}.csv'),
    )
    if brew_arg_overrides:
        job['brew_arg_overrides'] = brew_arg_overrides
    return job


def _dual_job(pair, repeat_slot, left_attacker, right_attacker, output_root, victim_seeds):
    pair_key = left_attacker['pair_key']
    pairing_id = f'{pair_key}_rep{repeat_slot + 1}'
    dual_dir = os.path.join(output_root, 'dual')
    return dict(
        job_id=f'dual_{pairing_id}',
        pairing_id=pairing_id,
        condition=f'{pair["target_class_name"]}_to_{pair["poison_class_name"]}_{pair["pair_bin"]}',
        distance_bucket=pair['pair_bin'],
        pair_key=pair_key,
        pair_number=int(pair['pair_number']),
        pair_bin=pair['pair_bin'],
        target_pair_left_index=int(pair['left_index']),
        target_pair_right_index=int(pair['right_index']),
        shared_target_class=int(pair['target_class']),
        shared_target_class_name=pair['target_class_name'],
        shared_poison_class=int(pair['poison_class']),
        shared_poison_class_name=pair['poison_class_name'],
        feature_cosine_distance=float(pair['feature_cosine_distance']),
        feature_cosine_similarity=float(pair['feature_cosine_similarity']),
        target_target_distance=float(pair['feature_cosine_distance']),
        gradient_cosine=float(pair['gradient_cosine']),
        attackers=[left_attacker, right_attacker],
        brew_artifact_paths=[
            os.path.join(output_root, 'brews', f'{left_attacker["brew_job_id"]}.pt'),
            os.path.join(output_root, 'brews', f'{right_attacker["brew_job_id"]}.pt'),
        ],
        victim_seeds=list(victim_seeds),
        overlap_seed=_pair_overlap_seed(pairing_id, 0),
        overlap_policy='assign_one_owner',
        arg_overrides=dict(name=f'dual_{pairing_id}', poisonkey=None),
        output_path=os.path.join(dual_dir, f'dual_{pairing_id}.csv'),
    )


def build_c6_experiment(
    *,
    experiment_id,
    output_root,
    common_args,
    scheduler,
    class_names,
    rankings,
    distance_matrix,
    pair_selections,
    pair_summaries,
    target_poison_pairs,
    repeats,
    victim_seeds,
    distance_artifact_path,
    clean_model_path,
):
    base_budget = float(common_args['budget'])
    brew_dir = os.path.join(output_root, 'brews')
    solo_dir = os.path.join(output_root, 'solo')
    target_rank_cache = {}

    brew_jobs = []
    solo_jobs = []
    dual_jobs = []
    seen_attacker_ids = set()

    for pair in pair_selections:
        target_class = int(pair['target_class'])
        poison_class = int(pair['poison_class'])
        if poison_class not in target_rank_cache:
            target_rank_cache[poison_class] = _source_target_ranks(class_names[poison_class], rankings)
        source_target_rank = target_rank_cache[poison_class][target_class]
        source_target_distance = float(distance_matrix[poison_class, target_class])

        for repeat_slot in range(repeats):
            standard_specs = []
            for role, target_index in (('A', pair['left_index']), ('B', pair['right_index'])):
                standard_spec = _build_attacker_spec(
                    pair=pair,
                    role=role,
                    target_index=target_index,
                    repeat_slot=repeat_slot,
                    budget_multiplier=1,
                    base_budget=base_budget,
                    source_target_distance=source_target_distance,
                    source_target_rank=source_target_rank,
                    brew_dir=brew_dir,
                )
                doubled_spec = _build_attacker_spec(
                    pair=pair,
                    role=role,
                    target_index=target_index,
                    repeat_slot=repeat_slot,
                    budget_multiplier=2,
                    base_budget=base_budget,
                    source_target_distance=source_target_distance,
                    source_target_rank=source_target_rank,
                    brew_dir=brew_dir,
                )
                for spec in (standard_spec, doubled_spec):
                    attacker_id = spec['attacker']['attacker_id']
                    if attacker_id in seen_attacker_ids:
                        raise ValueError(f'Duplicate attacker id generated: {attacker_id}')
                    seen_attacker_ids.add(attacker_id)
                    brew_jobs.append(spec)
                    solo_jobs.append(_solo_job(spec, solo_dir, victim_seeds))
                standard_specs.append(standard_spec)

            dual_jobs.append(_dual_job(
                pair,
                repeat_slot,
                standard_specs[0]['attacker'],
                standard_specs[1]['attacker'],
                output_root,
                victim_seeds,
            ))

    scheduler_payload = dict(scheduler or {})
    per_stage = dict(scheduler_payload.get('per_stage', {}))
    per_stage.update(SCHEDULER_PER_STAGE_OVERRIDES)
    scheduler_payload['per_stage'] = per_stage

    return dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        family='C6',
        class_names=list(class_names),
        metadata=dict(
            repeats=repeats,
            victim_seeds=list(victim_seeds),
            target_poison_pairs=[
                dict(target=target_name, poison=poison_name)
                for target_name, poison_name in target_poison_pairs
            ],
            pair_bins=[dict(bin_spec) for bin_spec in PAIR_BINS],
            pairs_per_bin=math.floor(len(pair_selections) / (len(target_poison_pairs) * len(PAIR_BINS))),
            pair_selection_strategy='within_class_clean_representation_cosine_distance',
            disjoint_target_images='within target class across all selected bins',
            doubled_budget_controls=True,
            base_budget=base_budget,
            doubled_budget=base_budget * 2,
            distance_artifact_path=distance_artifact_path,
            clean_model_path=clean_model_path,
            selected_pair_summaries=pair_summaries,
        ),
        scheduler=scheduler_payload,
        common_args=common_args,
        output_root=output_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )


def _parse_target_poison_pairs(raw_value):
    if raw_value.strip() == '':
        return list(TARGET_POISON_PAIRS)
    pairs = []
    for item in raw_value.split(','):
        target, poison = item.split(':')
        pairs.append((target.strip(), poison.strip()))
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    shared_planner_defaults = planner_defaults()
    parser.add_argument(
        '--distance_artifact',
        default='artifacts/c1/class_distance_matrix_cifar10_resnet18_modelkey0.pt',
        type=str,
        help='Saved class-distance matrix artifact with clean-model metadata.',
    )
    parser.add_argument('--output', default='artifacts/c6/c6_target_pair_similarity.json', type=str)
    parser.add_argument('--pairs-per-bin', default=5, type=int)
    parser.add_argument('--repeats', default=8, type=int)
    parser.add_argument('--target-poison-pairs', default='', type=str,
                        help='Comma-separated target:poison pairs. Defaults to airplane:dog,bird:truck,cat:airplane.')
    parser.add_argument('--train-clean-if-missing', action='store_true',
                        help='Train the clean model during preparation if the distance artifact cache is missing.')
    parser.add_argument('--clean-model-path', default=None, type=str,
                        help='Override the clean ResNet checkpoint path stored in the distance artifact.')
    args = parser.parse_args()

    common_args = common_args_defaults()
    common_args['eps'] = EPS_OVERRIDE

    distance_artifact_path = os.path.expanduser(args.distance_artifact)
    distance_artifact = torch.load(distance_artifact_path, map_location='cpu')
    class_names = list(distance_artifact['class_names'])
    target_poison_pairs = _parse_target_poison_pairs(args.target_poison_pairs)
    for target_name, poison_name in target_poison_pairs:
        _class_index(class_names, target_name)
        _class_index(class_names, poison_name)

    setup = forest.utils.system_startup(build_args_namespace(common_args))
    model, clean_model_path = _load_clean_model(
        common_args,
        distance_artifact,
        setup,
        clean_model_override=args.clean_model_path,
        train_if_missing=args.train_clean_if_missing,
    )
    batch_size = int(distance_artifact.get('batch_size') or 128)
    _trainset, validset = construct_datasets(common_args['dataset'], common_args['data_path'], normalize=True)
    class_to_valid_indices = class_to_dataset_indices(validset)
    experiment_path, experiment_id, output_root = output_paths_from_output_arg(args.output)

    pair_selections, pair_summaries = _target_pair_selections(
        model=model,
        validset=validset,
        class_to_valid_indices=class_to_valid_indices,
        class_names=class_names,
        target_poison_pairs=target_poison_pairs,
        pairs_per_bin=args.pairs_per_bin,
        batch_size=batch_size,
        setup=setup,
    )
    experiment = build_c6_experiment(
        experiment_id=experiment_id,
        output_root=output_root,
        common_args=common_args,
        scheduler=scheduler_defaults(),
        class_names=class_names,
        rankings=distance_artifact['rankings'],
        distance_matrix=distance_artifact['distance_matrix'],
        pair_selections=pair_selections,
        pair_summaries=pair_summaries,
        target_poison_pairs=target_poison_pairs,
        repeats=args.repeats,
        victim_seeds=shared_planner_defaults['victim_seeds'],
        distance_artifact_path=distance_artifact_path,
        clean_model_path=clean_model_path,
    )
    save_experiment(experiment, experiment_path)
    print(
        f'Saved C6 target-pair similarity spec to {experiment_path} '
        f'(pairs={len(pair_selections)}, repeats={args.repeats}, '
        f'brews={len(experiment["brew_jobs"])}, solos={len(experiment["solo_jobs"])}, '
        f'duals={len(experiment["dual_jobs"])}, eps={common_args["eps"]}, '
        f'budget={common_args["budget"]}, victim_seeds={shared_planner_defaults["victim_seeds"]}).'
    )
