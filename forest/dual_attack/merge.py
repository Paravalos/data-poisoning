"""Poison merge helpers for dual-attacker runs."""

from __future__ import annotations

import random

import torch


def coerce_index_list(value):
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def resolve_dual_overlap(left, right, overlap_seed):
    left_indices = coerce_index_list(left['poison_indices'])
    right_indices = coerce_index_list(right['poison_indices'])
    left_lookup = {idx: pos for pos, idx in enumerate(left_indices)}
    right_lookup = {idx: pos for pos, idx in enumerate(right_indices)}

    shared = sorted(set(left_indices) & set(right_indices))
    keep_left = set(left_indices) - set(shared)
    keep_right = set(right_indices) - set(shared)
    lost = {
        left['attacker']['attacker_id']: 0,
        right['attacker']['attacker_id']: 0,
    }

    rng = random.Random(overlap_seed)
    for index in shared:
        if rng.random() < 0.5:
            keep_left.add(index)
            lost[right['attacker']['attacker_id']] += 1
        else:
            keep_right.add(index)
            lost[left['attacker']['attacker_id']] += 1

    merged_indices = sorted(keep_left | keep_right)
    merged_delta = torch.stack([
        left['poison_delta'][left_lookup[index]] if index in keep_left else right['poison_delta'][right_lookup[index]]
        for index in merged_indices
    ])

    return merged_indices, merged_delta, dict(
        overlap_total=len(shared),
        lost_by_attacker=lost,
        overlap_seed=overlap_seed,
    )
