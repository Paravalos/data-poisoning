"""Poison merge helpers for dual-attacker runs."""

from __future__ import annotations

import random

import torch


def coerce_index_list(value):
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def _sum_clipped_bounds(reference_delta, eps, data_std):
    if eps is None or data_std is None:
        raise ValueError('sum_clipped overlap resolution requires eps and data_std.')
    std = torch.as_tensor(data_std, dtype=reference_delta.dtype, device=reference_delta.device)
    view_shape = [1, std.shape[0]] + [1] * (reference_delta.dim() - 2)
    return (float(eps) / 255.0 / std.view(*view_shape))


def resolve_dual_overlap(left, right, overlap_seed, merge_rule='assign_one_owner', eps=None, data_std=None):
    left_indices = coerce_index_list(left['poison_indices'])
    right_indices = coerce_index_list(right['poison_indices'])
    left_lookup = {idx: pos for pos, idx in enumerate(left_indices)}
    right_lookup = {idx: pos for pos, idx in enumerate(right_indices)}

    shared = sorted(set(left_indices) & set(right_indices))
    lost = {
        left['attacker']['attacker_id']: 0,
        right['attacker']['attacker_id']: 0,
    }

    if merge_rule == 'assign_one_owner':
        keep_left = set(left_indices) - set(shared)
        keep_right = set(right_indices) - set(shared)
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
    elif merge_rule == 'sum_clipped':
        merged_indices = sorted(set(left_indices) | set(right_indices))
        clip_bounds = _sum_clipped_bounds(left['poison_delta'], eps, data_std)
        merged_delta = torch.stack([
            torch.clamp(
                left['poison_delta'][left_lookup[index]] + right['poison_delta'][right_lookup[index]],
                min=-clip_bounds[0],
                max=clip_bounds[0],
            )
            if index in left_lookup and index in right_lookup
            else left['poison_delta'][left_lookup[index]] if index in left_lookup
            else right['poison_delta'][right_lookup[index]]
            for index in merged_indices
        ])
    else:
        raise ValueError(f'Unsupported merge_rule {merge_rule}.')

    return merged_indices, merged_delta, dict(
        overlap_total=len(shared),
        lost_by_attacker=lost,
        overlap_seed=overlap_seed,
        merge_rule=merge_rule,
        effective_unique_poison_count=len(merged_indices),
    )
