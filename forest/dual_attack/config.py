"""YAML-backed shared defaults for dual-attack experiment preparation."""

from __future__ import annotations

import os
from functools import lru_cache

import forest
import yaml


DEFAULTS_YAML_PATH = os.path.join(os.path.dirname(__file__), 'defaults.yaml')


@lru_cache(maxsize=1)
def load_defaults():
    with open(DEFAULTS_YAML_PATH, 'r') as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def _defaults_section(section_name):
    return dict(load_defaults().get(section_name, {}))


def common_args_defaults():
    defaults = vars(forest.options().parse_args([]))
    defaults.update(_defaults_section('common_args'))
    return defaults


def planner_defaults():
    return _defaults_section('planner_defaults')


def c1_planner_defaults():
    defaults = planner_defaults()
    defaults.update(load_defaults().get('family_defaults', {}).get('c1', {}))
    return defaults


def scheduler_defaults():
    return _defaults_section('scheduler_defaults')
