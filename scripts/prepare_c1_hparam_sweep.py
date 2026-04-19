"""Prepare a small C1 eps/budget sweep from an existing C1 experiment JSON."""

from __future__ import annotations

import argparse
import copy
import json
import os

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)


def _parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(',') if item.strip() != '']


def _parse_int_list(raw_value):
    return [int(item.strip()) for item in raw_value.split(',') if item.strip() != '']


def _class_index(class_names, class_name_or_index):
    if class_name_or_index.isdigit():
        return int(class_name_or_index)
    if class_name_or_index not in class_names:
        raise ValueError(f'Unknown class {class_name_or_index}. Choices: {class_names}.')
    return class_names.index(class_name_or_index)


def _format_value(value):
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def _load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def _save_json(payload, path):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _matching_attackers(base_experiment, source_class, target_class, selection_keys):
    seen = set()
    attackers = []
    for job in base_experiment.get('brew_jobs', []):
        attacker = job['attacker']
        key = (attacker['source_class'], attacker['target_adv_class'], attacker['selection_key'])
        if key in seen:
            continue
        if int(attacker['source_class']) != source_class:
            continue
        if int(attacker['target_adv_class']) != target_class:
            continue
        if selection_keys is not None and int(attacker['selection_key']) not in selection_keys:
            continue
        seen.add(key)
        attackers.append(copy.deepcopy(attacker))
    return sorted(attackers, key=lambda attacker: int(attacker['selection_key']))


def build_sweep_experiment(
    *,
    base_experiment,
    experiment_id,
    output_root,
    source_class,
    target_class,
    selection_keys,
    eps_values,
    budget_values,
    victim_seeds,
):
    attackers = _matching_attackers(base_experiment, source_class, target_class, selection_keys)
    if len(attackers) == 0:
        raise ValueError('No matching base attackers found. Check source/target class and selection keys.')

    common_args = copy.deepcopy(base_experiment.get('common_args', {}))
    brew_dir = os.path.join(output_root, 'brews')
    solo_dir = os.path.join(output_root, 'solo')

    brew_jobs = []
    solo_jobs = []
    sampled_target_indices = {}

    for eps in eps_values:
        for budget in budget_values:
            sweep_suffix = f'eps{_format_value(eps)}_budget{_format_value(budget)}'
            for base_attacker in attackers:
                attacker = copy.deepcopy(base_attacker)
                base_attacker_id = attacker['attacker_id']
                attacker_id = f'{base_attacker_id}_{sweep_suffix}'
                brew_job_id = f'brew_{attacker_id}'
                artifact_path = os.path.join(brew_dir, f'{brew_job_id}.pt')
                brew_overrides = dict(
                    poisonkey=attacker['poisonkey'],
                    name=brew_job_id,
                    targets=1,
                    eps=eps,
                    budget=budget,
                )
                attacker.update(dict(
                    attacker_id=attacker_id,
                    base_attacker_id=base_attacker_id,
                    brew_job_id=brew_job_id,
                    eps=eps,
                    budget=budget,
                ))
                sampled_target_indices.setdefault(str(attacker['source_class']), []).append(int(attacker['target_index']))

                brew_jobs.append(dict(
                    job_id=brew_job_id,
                    attacker=attacker,
                    arg_overrides=brew_overrides,
                    artifact_path=artifact_path,
                ))
                solo_jobs.append(dict(
                    job_id=f'solo_{attacker_id}',
                    attacker=attacker,
                    brew_artifact_path=artifact_path,
                    brew_arg_overrides=dict(eps=eps, budget=budget),
                    victim_seeds=list(victim_seeds),
                    arg_overrides=dict(name=f'solo_{attacker_id}', poisonkey=None, eps=eps, budget=budget),
                    output_path=os.path.join(solo_dir, f'solo_{attacker_id}.csv'),
                ))

    return dict(
        schema_version=base_experiment['schema_version'],
        experiment_id=experiment_id,
        family=base_experiment.get('family', 'C1'),
        class_names=list(base_experiment['class_names']),
        metadata=dict(
            shared_target_class=target_class,
            fixed_attacker_a_source_class=source_class,
            planning_seed=base_experiment.get('metadata', {}).get('planning_seed', ''),
            repeats=len(attackers),
            victim_seeds=list(victim_seeds),
            source_experiment_id=base_experiment.get('experiment_id'),
            pair_selection_strategy='single_source_eps_budget_sweep',
            eps_values=list(eps_values),
            budget_values=list(budget_values),
            selection_keys=[int(attacker['selection_key']) for attacker in attackers],
            sampled_target_indices=sampled_target_indices,
        ),
        scheduler=copy.deepcopy(base_experiment.get('scheduler', {})),
        common_args=common_args,
        output_root=output_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=[],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a disposable C1 eps/budget sweep experiment.')
    parser.add_argument('--base-experiment', default='artifacts/c1/c1_experiment.json', type=str)
    parser.add_argument('--output', default='artifacts/c1/c1_dog_to_airplane_hparam_sweep.json', type=str)
    parser.add_argument('--source-class', default='dog', type=str)
    parser.add_argument('--target-class', default='airplane', type=str)
    parser.add_argument('--selection-keys', default='0', type=str,
                        help='Comma-separated target selection keys from the base C1 JSON. Use empty string for all matches.')
    parser.add_argument('--eps-values', default='16,8,4', type=str)
    parser.add_argument('--budget-values', default='0.01,0.005,0.0025,0.001', type=str)
    parser.add_argument('--victim-seeds', default='0,1,2', type=str)
    parser.add_argument('--output-root', default=None, type=str)
    args = parser.parse_args()

    base_path = os.path.expanduser(args.base_experiment)
    output_path = os.path.expanduser(args.output)
    base_experiment = _load_json(base_path)
    experiment_id = os.path.splitext(os.path.basename(output_path))[0]
    output_root = os.path.expanduser(args.output_root) if args.output_root else os.path.join(
        os.path.dirname(output_path),
        experiment_id,
    )

    selection_keys = None if args.selection_keys.strip() == '' else _parse_int_list(args.selection_keys)
    source_class = _class_index(base_experiment['class_names'], args.source_class)
    target_class = _class_index(base_experiment['class_names'], args.target_class)

    experiment = build_sweep_experiment(
        base_experiment=base_experiment,
        experiment_id=experiment_id,
        output_root=output_root,
        source_class=source_class,
        target_class=target_class,
        selection_keys=selection_keys,
        eps_values=_parse_float_list(args.eps_values),
        budget_values=_parse_float_list(args.budget_values),
        victim_seeds=_parse_int_list(args.victim_seeds),
    )
    _save_json(experiment, output_path)
    print(f'Saved C1 hyperparameter sweep spec to {output_path}.')
    print(f'Brew jobs: {len(experiment["brew_jobs"])}')
    print(f'Solo jobs: {len(experiment["solo_jobs"])}')
