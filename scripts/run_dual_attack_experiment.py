"""Run brew, solo, or dual jobs from a prepared dual-attacker experiment JSON."""

from __future__ import annotations

import argparse

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.experiment import iter_stage_jobs, load_experiment
from forest.dual_attack.runtime import run_brew_job, run_dual_job, run_solo_job


def _print_commands(experiment_path, selected_jobs):
    for stage_name, job in selected_jobs:
        print(
            f'python scripts/run_dual_attack_experiment.py --experiment {experiment_path} '
            f'--stage {stage_name} --job-id {job["job_id"]}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run jobs from a dual-attacker experiment JSON.')
    parser.add_argument('--experiment', required=True, type=str, help='Path to the experiment JSON.')
    parser.add_argument('--stage', default='all', choices=['brew', 'solo', 'dual', 'all'])
    parser.add_argument('--job-id', default=None, type=str, help='Optional single job id to run.')
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--print-commands', action='store_true', help='Print stage/job commands instead of running them.')
    args = parser.parse_args()

    experiment = load_experiment(args.experiment)
    selected_jobs = [
        (stage_name, job)
        for stage_name, job in iter_stage_jobs(experiment, args.stage)
        if args.job_id is None or job['job_id'] == args.job_id
    ]

    if args.print_commands:
        _print_commands(args.experiment, selected_jobs)
    else:
        for stage_name, job in selected_jobs:
            if stage_name == 'brew':
                run_brew_job(experiment, job, force_dryrun=args.dryrun)
            elif stage_name == 'solo':
                run_solo_job(experiment, job, force_dryrun=args.dryrun)
            elif stage_name == 'dual':
                run_dual_job(experiment, job, force_dryrun=args.dryrun)
