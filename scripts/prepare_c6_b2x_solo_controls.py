"""Generate C6 double-budget solo-control jobs from an existing C6 experiment.

The pulled C6 result JSON contains the normal-budget dual and solo runs. This
script derives the missing b2x solo-control manifest without changing the
original experiment:

  * one b2x brew spec for every existing b1x attacker
  * one b2x solo eval job for every derived b2x brew
  * no dual jobs

If the b2x brew artifacts already exist, submit the derived manifest with
``--stage solo --skip-completed`` and the submitter will treat the brew
dependencies as completed.
"""

from __future__ import annotations

import argparse
import copy
import os

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.experiment import load_experiment, save_experiment


def _replace_suffix(value, old='_b1x', new='_b2x'):
    if not isinstance(value, str):
        return value
    if not value.endswith(old):
        raise ValueError(f'Expected value to end with {old!r}: {value}')
    return value[: -len(old)] + new


def _replace_b1x_tokens(value):
    if not isinstance(value, str):
        return value
    return value.replace('_b1x', '_b2x')


def _b2x_attacker(attacker, base_budget):
    updated = copy.deepcopy(attacker)
    updated['attacker_id'] = _replace_suffix(updated['attacker_id'])
    updated['brew_job_id'] = f'brew_{updated["attacker_id"]}'
    updated['budget_multiplier'] = 2
    updated['base_budget'] = float(base_budget)
    updated['effective_budget'] = float(base_budget) * 2.0
    return updated


def _b2x_brew_job(job, base_budget):
    attacker = _b2x_attacker(job['attacker'], base_budget)
    updated = copy.deepcopy(job)
    updated['job_id'] = attacker['brew_job_id']
    updated['attacker'] = attacker
    updated['artifact_path'] = _replace_b1x_tokens(updated['artifact_path'])
    overrides = dict(updated.get('arg_overrides') or {})
    overrides.update(
        name=attacker['brew_job_id'],
        poisonkey=attacker['poisonkey'],
        poison_ids_seed=attacker.get('poison_ids_seed'),
        targets=1,
        budget=attacker['effective_budget'],
    )
    updated['arg_overrides'] = overrides
    return updated


def _b2x_solo_job(job, base_budget):
    attacker = _b2x_attacker(job['attacker'], base_budget)
    updated = copy.deepcopy(job)
    updated['job_id'] = f'solo_{attacker["attacker_id"]}'
    updated['attacker'] = attacker
    updated['brew_artifact_path'] = _replace_b1x_tokens(updated['brew_artifact_path'])
    updated['output_path'] = _replace_b1x_tokens(updated['output_path'])
    updated['arg_overrides'] = dict(
        name=updated['job_id'],
        poisonkey=None,
        budget=attacker['effective_budget'],
    )
    updated['brew_arg_overrides'] = dict(budget=attacker['effective_budget'])
    return updated


def build_b2x_manifest(source):
    metadata = dict(source.get('metadata') or {})
    base_budget = float(metadata.get('base_budget', source.get('common_args', {}).get('budget', 0.01)))
    output_root = source['output_root']

    brew_jobs = [
        _b2x_brew_job(job, base_budget)
        for job in source.get('brew_jobs', [])
        if job.get('attacker', {}).get('budget_multiplier') in (None, 1)
    ]
    solo_jobs = [
        _b2x_solo_job(job, base_budget)
        for job in source.get('solo_jobs', [])
        if job.get('attacker', {}).get('budget_multiplier') in (None, 1)
    ]

    metadata.update(
        doubled_budget_controls=True,
        generated_from_experiment_id=source.get('experiment_id'),
        generated_control='solo_b2x',
        base_budget=base_budget,
        doubled_budget=base_budget * 2.0,
    )

    return dict(
        schema_version=source['schema_version'],
        experiment_id=f'{source["experiment_id"]}_b2x_solo_controls',
        family=source.get('family', 'C6'),
        class_names=list(source['class_names']),
        metadata=metadata,
        scheduler=copy.deepcopy(source.get('scheduler') or {}),
        common_args=copy.deepcopy(source.get('common_args') or {}),
        output_root=output_root,
        brew_jobs=brew_jobs,
        solo_jobs=solo_jobs,
        dual_jobs=[],
    )


def default_output_path(source_path):
    root, ext = os.path.splitext(os.path.expanduser(source_path))
    return f'{root}_b2x_solo_controls{ext or ".json"}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        default='artifacts/c6/c6_target_pair_similarity.json',
        type=str,
        help='Existing C6 experiment JSON with b1x jobs.',
    )
    parser.add_argument(
        '--output',
        default=None,
        type=str,
        help='Where to write the derived b2x solo-control experiment JSON.',
    )
    args = parser.parse_args()

    source_path = os.path.expanduser(args.source)
    output_path = os.path.expanduser(args.output or default_output_path(source_path))
    source = load_experiment(source_path)
    manifest = build_b2x_manifest(source)
    save_experiment(manifest, output_path)
    print(
        f'Wrote {output_path} '
        f'(brews={len(manifest["brew_jobs"])}, solos={len(manifest["solo_jobs"])}, duals=0).'
    )
    print('Submit existing-brew solo controls with:')
    print(f'  python scripts/submit_dual_attack_experiment.py --experiment {output_path} --stage solo')
    print('Or run one local job with:')
    print(
        '  python scripts/run_dual_attack_experiment.py '
        f'--experiment {output_path} --stage solo --job-id {manifest["solo_jobs"][0]["job_id"]}'
    )


if __name__ == '__main__':
    main()
