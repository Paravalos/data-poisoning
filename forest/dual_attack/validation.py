"""Validation helpers for artifact-backed dual-attacker runs."""

from __future__ import annotations

import torch

from .experiment import brew_identity_payload, build_args_namespace


def _normalize_value(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _assert_equal(field_name, expected, actual):
    if _normalize_value(expected) != _normalize_value(actual):
        raise ValueError(f'Artifact mismatch for {field_name}: expected {expected}, found {actual}.')


def _artifact_brew_identity(artifact):
    identity = artifact.get('brew_identity')
    if identity is not None:
        return identity
    brew_args = artifact.get('brew_config', {}).get('args', {})
    return brew_identity_payload(build_args_namespace(common_args=brew_args), artifact.get('attacker', {}))


def validate_brew_identity(artifact, expected_attacker, expected_args):
    _assert_equal('brew_identity', brew_identity_payload(expected_args, expected_attacker), _artifact_brew_identity(artifact))


def validate_brew_artifact(experiment, artifact, expected_attacker, expected_args):
    validate_brew_identity(artifact, expected_attacker, expected_args)

    attacker = artifact.get('attacker', {})
    for field in (
        'attacker_id',
        'brew_job_id',
        'repeat_slot',
        'target_index',
        'source_class',
        'target_true_class',
        'target_adv_class',
        'poison_ids_seed',
        'explicit_poison_ids',
        'bucket',
    ):
        expected_value = expected_attacker.get(field)
        actual_value = attacker.get(field)
        if expected_value is None and actual_value is None:
            continue
        _assert_equal(f'attacker.{field}', expected_value, actual_value)

    _assert_equal('target_index', expected_attacker['target_index'], artifact.get('target_index'))
    _assert_equal('source_class', expected_attacker['source_class'], artifact.get('source_class'))
    _assert_equal('target_true_class', expected_attacker['target_true_class'], artifact.get('target_true_class'))
    _assert_equal('target_adv_class', expected_attacker['target_adv_class'], artifact.get('target_adv_class'))
