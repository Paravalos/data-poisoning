"""Submit dual-attacker experiment jobs to Slurm with dependencies."""

from __future__ import annotations

import argparse
import datetime
import os
import sys

from _dual_attack_script_utils import ensure_repo_root

REPO_ROOT = ensure_repo_root(__file__)

from forest.dual_attack.experiment import load_experiment
from forest.dual_attack.summary import summarize_experiment
from forest.dual_attack.submitter import (
    build_sbatch_command,
    default_submission_log_path,
    load_submission_log,
    plan_submission_specs,
    save_submission_log,
    submit_sbatch_command,
)


def _output_dir(experiment, experiment_path, output_dir_override):
    if output_dir_override is not None:
        return os.path.expanduser(output_dir_override)
    experiment_root = experiment.get('output_root')
    if experiment_root:
        return os.path.join(experiment_root, 'slurm')
    return os.path.join(os.path.dirname(os.path.expanduser(experiment_path)), 'slurm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit dual-attacker experiment jobs to Slurm.')
    parser.add_argument('--experiment', required=True, type=str, help='Path to the experiment JSON.')
    parser.add_argument('--stage', default='all', choices=['brew', 'solo', 'dual', 'all'])
    parser.add_argument('--job-id', default=None, type=str, help='Optional single job id to submit.')
    parser.add_argument('--dual-dependency-stage', default='solo', choices=['solo', 'brew'],
                        help='What dual jobs should depend on.')
    parser.add_argument('--print-only', action='store_true', help='Print sbatch commands instead of submitting them.')
    parser.add_argument('--print-summary', action='store_true', help='Print a human-friendly summary of the experiment JSON and exit.')
    parser.add_argument('--submission-log', default=None, type=str, help='Where to store/load submitted Slurm ids.')
    parser.add_argument('--output-dir', default=None, type=str, help='Directory for Slurm stdout/stderr files.')
    parser.add_argument('--python-executable', default=sys.executable, type=str)
    parser.add_argument('--account', default=None, type=str)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--mem', default=None, type=str)
    parser.add_argument('--cpus', default=None, type=int)
    parser.add_argument('--time', default=None, type=str)
    args = parser.parse_args()

    experiment_path = os.path.expanduser(args.experiment)
    experiment = load_experiment(experiment_path)
    if args.print_summary:
        print(summarize_experiment(experiment))
        raise SystemExit(0)

    submission_log_path = os.path.expanduser(
        args.submission_log or default_submission_log_path(experiment_path)
    )
    submission_log = load_submission_log(submission_log_path)
    submitted_job_ids = submission_log.setdefault('submissions', {})
    printed_commands = submission_log.setdefault('printed_commands', {})
    output_dir = _output_dir(experiment, experiment_path, args.output_dir)

    specs = plan_submission_specs(
        experiment,
        experiment_path,
        args.stage,
        args.dual_dependency_stage,
        REPO_ROOT,
        python_executable=args.python_executable,
    )
    if args.job_id is not None:
        specs = [spec for spec in specs if spec['job_id'] == args.job_id]

    for spec in specs:
        dependency_slurm_ids = []
        for dependency_job_id in spec['dependency_job_ids']:
            slurm_id = submitted_job_ids.get(dependency_job_id)
            if slurm_id is None:
                if args.print_only:
                    placeholder = f'${dependency_job_id.upper()}_SLURM_ID'
                    dependency_slurm_ids.append(placeholder)
                else:
                    raise ValueError(
                        f'Cannot submit {spec["job_id"]}: missing Slurm id for dependency {dependency_job_id}. '
                        f'Submit dependencies first or use --stage all.'
                    )
            else:
                dependency_slurm_ids.append(str(slurm_id))

        sbatch_command = build_sbatch_command(spec, experiment, args, dependency_slurm_ids, output_dir)
        slurm_id, printed_command = submit_sbatch_command(sbatch_command, print_only=args.print_only)
        printed_commands[spec['job_id']] = printed_command
        if args.print_only:
            print(printed_command)
        else:
            submitted_job_ids[spec['job_id']] = slurm_id
            print(f'{spec["job_id"]}: {slurm_id}')

    submission_log.update(dict(
        experiment_path=experiment_path,
        updated_at=datetime.datetime.utcnow().isoformat() + 'Z',
        stage=args.stage,
        dual_dependency_stage=args.dual_dependency_stage,
        print_only=args.print_only,
        output_dir=output_dir,
    ))
    save_submission_log(submission_log_path, submission_log)
    print(f'Wrote submission log to {submission_log_path}.')
