"""Custom kettle implementation for artifact-driven dual-attacker evaluation."""

from __future__ import annotations

import torch

from forest.data.datasets import Subset
from forest.data.kettle_base import _Kettle


class ArtifactKettle(_Kettle):
    """Kettle variant backed by saved brew artifacts instead of a poisonkey."""

    def __init__(self, args, batch_size, augmentations, mixing_method, attack_artifacts, merged_indices, setup):
        self.attack_artifacts = attack_artifacts
        self.merged_indices = [int(idx) for idx in merged_indices]
        super().__init__(args, batch_size, augmentations, mixing_method, setup=setup)

    def prepare_experiment(self):
        intended_classes = [int(artifact['target_adv_class']) for artifact in self.attack_artifacts]
        target_indices = [int(artifact['target_index']) for artifact in self.attack_artifacts]

        self.poison_setup = dict(
            poison_budget=self.args.budget,
            target_num=len(self.attack_artifacts),
            poison_class=intended_classes[0] if len(set(intended_classes)) == 1 else None,
            target_class=int(self.attack_artifacts[0]['target_true_class']),
            intended_class=intended_classes,
        )

        self.poison_ids = torch.as_tensor(self.merged_indices, dtype=torch.long)
        self.poisonset = Subset(self.trainset, indices=self.merged_indices)
        self.target_ids = target_indices
        self.targetset = Subset(self.validset, indices=target_indices)

        held_out = set(target_indices)
        valid_indices = [idx for idx in range(len(self.validset)) if idx not in held_out]
        self.validset = Subset(self.validset, indices=valid_indices)
        self.poison_lookup = dict(zip(self.merged_indices, range(len(self.merged_indices))))
