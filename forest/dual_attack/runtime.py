"""Runtime entrypoints for dual-attacker brewing and evaluation."""

from __future__ import annotations

import datetime
import time

import forest
import torch

from forest.dual_attack.artifacts import compute_delta_norm_summary, load_brew_artifact, save_brew_artifact, write_rows
from forest.dual_attack.eval import evaluate_job
from forest.dual_attack.experiment import build_args_namespace
from forest.dual_attack.merge import coerce_index_list, resolve_dual_overlap
from forest.dual_attack.validation import validate_brew_artifact


def _job_args(experiment, job, force_dryrun=False):
    args = build_args_namespace(experiment.get('common_args'), job.get('arg_overrides'))
    if force_dryrun:
        args.dryrun = True
    if args.deterministic:
        forest.utils.set_deterministic()
    return args


def _expected_brew_args(experiment, attacker):
    return build_args_namespace(
        experiment.get('common_args'),
        dict(
            poisonkey=attacker['poisonkey'],
            poison_ids_seed=attacker.get('poison_ids_seed'),
            name=attacker['brew_job_id'],
            targets=1,
        ),
    )


def _system_startup(args):
    return forest.utils.system_startup(args)


def _timedelta_string(seconds):
    return str(datetime.timedelta(seconds=seconds)).replace(',', '')


def _artifact_poison_indices(poison_ids):
    return torch.as_tensor(poison_ids, dtype=torch.long).cpu()


def _validate_dual_artifacts(left, right):
    left_args = left['brew_config']['args']
    right_args = right['brew_config']['args']
    for field in ('dataset', 'scenario', 'threatmodel', 'recipe'):
        if left_args.get(field) != right_args.get(field):
            raise ValueError(f'Cannot merge brew artifacts with different {field}: {left_args.get(field)} vs {right_args.get(field)}.')
    if left_args.get('net') != right_args.get('net'):
        raise ValueError('Cannot merge brew artifacts brewed with different network settings.')


def run_brew_job(experiment, job, force_dryrun=False):
    args = _job_args(experiment, job, force_dryrun=force_dryrun)
    setup = _system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    witch = forest.Witch(args, setup=setup)
    witch.patch_targets(data)

    start_time = time.time()
    if args.pretrained_model or args.skip_clean_training:
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    target_image, _, _ = data.targetset[0]
    artifact = dict(
        schema_version=1,
        experiment_id=experiment['experiment_id'],
        family=experiment['family'],
        job_id=job['job_id'],
        attacker=job['attacker'],
        poison_delta=poison_delta.detach().cpu(),
        poison_indices=_artifact_poison_indices(data.poison_ids),
        target_index=int(job['attacker']['target_index']),
        target_image=target_image.detach().cpu(),
        target_true_class=int(job['attacker']['target_true_class']),
        target_true_class_name=data.trainset.classes[int(job['attacker']['target_true_class'])],
        target_adv_class=int(job['attacker']['target_adv_class']),
        target_adv_class_name=data.trainset.classes[int(job['attacker']['target_adv_class'])],
        source_class=int(job['attacker']['source_class']),
        source_class_name=data.trainset.classes[int(job['attacker']['source_class'])],
        brew_config=dict(
            args={key: value for key, value in vars(args).items()},
            model_init_seed=model.model_init_seed,
            class_names=list(data.trainset.classes),
            clean_model_cache_path=model._compute_clean_model_cache_path()
            if hasattr(model, '_compute_clean_model_cache_path') else None,
        ),
        brew_loss=None if getattr(witch, 'stat_optimal_loss', None) is None else float(witch.stat_optimal_loss),
        poison_delta_norm=compute_delta_norm_summary(poison_delta.detach().cpu()),
        timestamps=dict(
            train_time=_timedelta_string(train_time - start_time),
            brew_time=_timedelta_string(brew_time - train_time),
        ),
        clean_stats_available=stats_clean is not None,
    )
    save_brew_artifact(job['artifact_path'], artifact)
    return dict(job_id=job['job_id'], artifact_path=job['artifact_path'])


def run_solo_job(experiment, job, force_dryrun=False):
    args = _job_args(experiment, job, force_dryrun=force_dryrun)
    artifact = load_brew_artifact(job['brew_artifact_path'])
    artifact['artifact_path'] = job['brew_artifact_path']
    validate_brew_artifact(experiment, artifact, job['attacker'], _expected_brew_args(experiment, job['attacker']))
    rows = evaluate_job(
        args,
        experiment,
        job,
        [artifact],
        artifact['poison_delta'],
        coerce_index_list(artifact['poison_indices']),
        dict(overlap_total=0, lost_by_attacker={artifact['attacker']['attacker_id']: 0}),
        run_type='solo',
        setup=_system_startup(args),
    )
    write_rows(job['output_path'], rows)
    return dict(job_id=job['job_id'], output_path=job['output_path'])


def run_dual_job(experiment, job, force_dryrun=False):
    args = _job_args(experiment, job, force_dryrun=force_dryrun)
    left = load_brew_artifact(job['brew_artifact_paths'][0])
    right = load_brew_artifact(job['brew_artifact_paths'][1])
    left['artifact_path'] = job['brew_artifact_paths'][0]
    right['artifact_path'] = job['brew_artifact_paths'][1]
    validate_brew_artifact(experiment, left, job['attackers'][0], _expected_brew_args(experiment, job['attackers'][0]))
    validate_brew_artifact(experiment, right, job['attackers'][1], _expected_brew_args(experiment, job['attackers'][1]))
    _validate_dual_artifacts(left, right)
    merged_indices, merged_delta, overlap_stats = resolve_dual_overlap(left, right, job['overlap_seed'])
    rows = evaluate_job(
        args,
        experiment,
        job,
        [left, right],
        merged_delta,
        merged_indices,
        overlap_stats,
        run_type='dual',
        setup=_system_startup(args),
    )
    write_rows(job['output_path'], rows)
    return dict(job_id=job['job_id'], output_path=job['output_path'])
