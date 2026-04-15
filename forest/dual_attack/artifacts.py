"""Artifact and result writing helpers for dual-attacker runs."""

from __future__ import annotations

import csv
import os

import torch


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)


def compute_delta_norm_summary(poison_delta):
    if poison_delta is None or len(poison_delta) == 0:
        return dict(mean_l2=0.0, max_l2=0.0, mean_linf=0.0, max_linf=0.0)

    flat = poison_delta.view(poison_delta.shape[0], -1)
    l2 = torch.linalg.norm(flat, dim=1)
    linf = flat.abs().amax(dim=1)
    return dict(
        mean_l2=float(l2.mean().item()),
        max_l2=float(l2.max().item()),
        mean_linf=float(linf.mean().item()),
        max_linf=float(linf.max().item()),
    )


def save_brew_artifact(path, artifact):
    ensure_parent_dir(path)
    torch.save(artifact, path)


def load_brew_artifact(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)


def write_rows(path, rows):
    ensure_parent_dir(path)
    if len(rows) == 0:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
