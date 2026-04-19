"""Slurm submission helpers for dual-attacker experiments."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

from forest.dual_attack.experiment import iter_stage_jobs, stage_to_job_key


def default_submission_log_path(experiment_path):
    base, _ = os.path.splitext(experiment_path)
    return f'{base}.submission_log.json'


def _job_output_path(stage_name, job):
    if stage_name == 'brew':
        return job.get('artifact_path')
    if stage_name in ('solo', 'dual'):
        return job.get('output_path')
    raise ValueError(f'Unsupported stage {stage_name}.')


def collect_completed_job_ids(experiment):
    """Return the set of job_ids whose on-disk output already exists."""
    completed = set()
    for stage_name in ('brew', 'solo', 'dual'):
        for job in experiment.get(stage_to_job_key(stage_name), []):
            path = _job_output_path(stage_name, job)
            if path and os.path.isfile(path):
                completed.add(job['job_id'])
    return completed


def runner_command(experiment_path, stage_name, job_id, repo_root, python_executable=None):
    if python_executable is None:
        python_executable = sys.executable
    runner_path = os.path.join(repo_root, 'scripts', 'run_dual_attack_experiment.py')
    return [
        python_executable,
        runner_path,
        '--experiment',
        experiment_path,
        '--stage',
        stage_name,
        '--job-id',
        job_id,
    ]


def _solo_dependency_job_ids(job):
    return [job['attacker']['brew_job_id']]


def _dual_dependency_job_ids(job, dependency_stage):
    if dependency_stage == 'solo':
        return [f"solo_{attacker['attacker_id']}" for attacker in job['attackers']]
    if dependency_stage == 'brew':
        return [attacker['brew_job_id'] for attacker in job['attackers']]
    raise ValueError(f'Unsupported dual dependency stage {dependency_stage}.')


def plan_submission_specs(experiment, experiment_path, stage, dual_dependency_stage, repo_root, python_executable=None):
    specs = []
    for stage_name, job in iter_stage_jobs(experiment, stage):
        if stage_name == 'brew':
            dependency_job_ids = []
        elif stage_name == 'solo':
            dependency_job_ids = _solo_dependency_job_ids(job)
        elif stage_name == 'dual':
            dependency_job_ids = _dual_dependency_job_ids(job, dual_dependency_stage)
        else:
            raise ValueError(f'Unsupported stage {stage_name}.')

        specs.append(dict(
            stage=stage_name,
            job_id=job['job_id'],
            dependency_job_ids=dependency_job_ids,
            command=runner_command(experiment_path, stage_name, job['job_id'], repo_root, python_executable),
        ))
    return specs


def _resolve_scheduler_value(experiment_scheduler, overrides, key, stage_name=None):
    override_value = getattr(overrides, key)
    if override_value is not None:
        return override_value
    if stage_name is not None:
        per_stage_value = experiment_scheduler.get('per_stage', {}).get(stage_name, {}).get(key)
        if per_stage_value is not None:
            return per_stage_value
    return experiment_scheduler.get(key)


def build_sbatch_command(spec, experiment, overrides, dependency_slurm_ids, output_dir):
    scheduler = experiment.get('scheduler', {})
    stage_name = spec.get('stage')
    account = _resolve_scheduler_value(scheduler, overrides, 'account', stage_name)
    gpu = _resolve_scheduler_value(scheduler, overrides, 'gpu', stage_name)
    mem = _resolve_scheduler_value(scheduler, overrides, 'mem', stage_name)
    cpus = _resolve_scheduler_value(scheduler, overrides, 'cpus', stage_name)
    time_limit = _resolve_scheduler_value(scheduler, overrides, 'time', stage_name)

    command = ['sbatch', '--parsable']
    if account:
        command.extend(['--account', str(account)])
    command.extend(['--nodes', '1'])
    if gpu:
        command.extend(['--gres', f'gpu:{gpu}:1'])
    if mem:
        command.extend(['--mem', str(mem)])
    if cpus:
        command.extend(['--cpus-per-task', str(cpus)])
    if time_limit:
        command.extend(['--time', str(time_limit)])

    os.makedirs(output_dir, exist_ok=True)
    command.extend(['--job-name', spec['job_id'][:128]])
    command.extend(['--output', os.path.join(output_dir, f'{spec["job_id"]}-%j.out')])
    command.extend(['--error', os.path.join(output_dir, f'{spec["job_id"]}-%j.err')])

    if len(dependency_slurm_ids) > 0:
        command.extend(['--dependency', f'afterok:{":".join(dependency_slurm_ids)}'])

    wrap_command = ' '.join(shlex.quote(part) for part in spec['command'])
    command.extend(['--wrap', wrap_command])
    return command


def format_shell_command(command):
    formatted = []
    for part in command:
        if '$' in part and ' ' not in part:
            formatted.append(part)
        else:
            formatted.append(shlex.quote(part))
    return ' '.join(formatted)


def submit_sbatch_command(command, print_only=False):
    if print_only:
        return None, format_shell_command(command)

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip(), format_shell_command(command)


def load_submission_log(path):
    if not os.path.isfile(path):
        return dict(submissions={})
    with open(path, 'r') as handle:
        return json.load(handle)


def save_submission_log(path, payload):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
