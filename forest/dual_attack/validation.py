"""Validation helpers for artifact-backed dual-attacker runs."""

from __future__ import annotations

import torch


BREW_IDENTITY_FIELDS = (
    'dataset',
    'net',
    'scenario',
    'threatmodel',
    'recipe',
    'budget',
    'eps',
    'attackoptim',
    'attackiter',
    'init',
    'tau',
    'restarts',
    'loss',
    'centreg',
    'normreg',
    'repel',
    'pbatch',
    'pshuffle',
    'paugment',
    'full_data',
    'ensemble',
    'stagger',
    'max_epoch',
    'targets',
    'patch_size',
    'data_aug',
    'randomize_deterministic_poison_ids',
)


def _normalize_value(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _assert_equal(field_name, expected, actual):
    if _normalize_value(expected) != _normalize_value(actual):
        raise ValueError(f'Artifact mismatch for {field_name}: expected {expected}, found {actual}.')


def validate_brew_artifact(experiment, artifact, expected_attacker, expected_args):
    attacker = artifact.get('attacker', {})
    for field in (
        'attacker_id',
        'selection_key',
        'repeat_slot',
        'target_index',
        'source_class',
        'target_true_class',
        'target_adv_class',
        'poison_ids_seed',
        'bucket',
    ):
        _assert_equal(f'attacker.{field}', expected_attacker.get(field), attacker.get(field))

    _assert_equal('target_index', expected_attacker['target_index'], artifact.get('target_index'))
    _assert_equal('source_class', expected_attacker['source_class'], artifact.get('source_class'))
    _assert_equal('target_true_class', expected_attacker['target_true_class'], artifact.get('target_true_class'))
    _assert_equal('target_adv_class', expected_attacker['target_adv_class'], artifact.get('target_adv_class'))

    brew_args = artifact.get('brew_config', {}).get('args', {})
    for field in BREW_IDENTITY_FIELDS:
        _assert_equal(f'brew_config.args.{field}', getattr(expected_args, field), brew_args.get(field))

    _assert_equal('brew_config.args.poisonkey', expected_args.poisonkey, brew_args.get('poisonkey'))
    _assert_equal('brew_config.args.poison_ids_seed', expected_args.poison_ids_seed, brew_args.get('poison_ids_seed'))
