"""JSON experiment schema helpers for dual-attacker runs."""

from __future__ import annotations

import json
import os

import forest


SCHEMA_VERSION = 1


def _normalize_list_option(value):
    if value is None or isinstance(value, list):
        return value
    return [value]


def build_args_namespace(common_args=None, arg_overrides=None):
    """Build a forest args namespace from experiment JSON fields."""
    parser = forest.options()
    args = parser.parse_args([])
    payload = vars(args)
    if common_args is not None:
        payload.update(common_args)
    if arg_overrides is not None:
        payload.update(arg_overrides)
    args.net = _normalize_list_option(args.net)
    args.vnet = _normalize_list_option(args.vnet)
    return args


def load_experiment(path):
    with open(path, 'r') as handle:
        payload = json.load(handle)
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported dual-attack schema version {payload.get("schema_version")}. '
            f'Expected {SCHEMA_VERSION}.'
        )
    return payload


def save_experiment(payload, path):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def stage_to_job_key(stage):
    mapping = dict(brew='brew_jobs', solo='solo_jobs', dual='dual_jobs')
    if stage not in mapping:
        raise ValueError(f'Unsupported stage {stage}.')
    return mapping[stage]


def iter_stage_jobs(experiment, stage):
    if stage == 'all':
        for stage_name in ('brew', 'solo', 'dual'):
            for job in experiment.get(stage_to_job_key(stage_name), []):
                yield stage_name, job
        return

    stage_key = stage_to_job_key(stage)
    for job in experiment.get(stage_key, []):
        yield stage, job
