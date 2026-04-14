"""Kettle variant: user pins target_class and/or poison_class via CLI.

target_id and poison image ids are sampled randomly under --poisonkey seeding.
This class is intentionally independent of KettleRandom so that future changes
to KettleRandom cannot alter fixed-class reproducibility.
"""
import warnings
import torch
import numpy as np

from .kettle_base import _Kettle
from ..utils import set_random_seed
from .datasets import Subset


class KettleFixedClass(_Kettle):

    def prepare_experiment(self):
        self.fixed_class_construction()

    def fixed_class_construction(self):
        if self.args.local_rank is None:
            if self.args.poisonkey is None:
                self.init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.init_seed = int(self.args.poisonkey)
            set_random_seed(self.init_seed)
            print(f'Initializing Poison data (chosen images, examples, targets, labels) '
                  f'with random seed {self.init_seed} '
                  f'[fixed: target_class={self.args.target_class}, poison_class={self.args.poison_class}]')
        else:
            rank = torch.distributed.get_rank()
            if self.args.poisonkey is None:
                init_seed = torch.randint(0, 2**32 - 1, [1], device=self.setup['device'])
            else:
                init_seed = torch.as_tensor(int(self.args.poisonkey), dtype=torch.int64,
                                            device=self.setup['device'])
            torch.distributed.broadcast(init_seed, src=0)
            if rank == 0:
                print(f'Initializing Poison data (chosen images, examples, targets, labels) '
                      f'with random seed {init_seed.item()} '
                      f'[fixed: target_class={self.args.target_class}, poison_class={self.args.poison_class}]')
            self.init_seed = init_seed.item()
            set_random_seed(self.init_seed)

        self.poison_setup = self._parse_threats_fixed_class()
        self.poisonset, self.targetset, self.validset = self._choose_poisons_randomly()

    def _parse_threats_fixed_class(self):
        num_classes = len(self.trainset.classes)
        fixed_t = self.args.target_class
        fixed_p = self.args.poison_class

        if fixed_t is not None and not (0 <= fixed_t < num_classes):
            raise ValueError(f'--target_class {fixed_t} out of range [0, {num_classes}).')
        if fixed_p is not None and not (0 <= fixed_p < num_classes):
            raise ValueError(f'--poison_class {fixed_p} out of range [0, {num_classes}).')

        target_class = fixed_t if fixed_t is not None else np.random.randint(num_classes)

        list_intentions = list(range(num_classes))
        if target_class is not None:
            list_intentions.remove(target_class)
        intended_class = [np.random.choice(list_intentions)] * self.args.targets

        if self.args.targets < 1:
            warnings.warn('Number of targets set to 0.')
            return dict(poison_budget=0, target_num=0,
                        poison_class=fixed_p if fixed_p is not None else np.random.randint(num_classes),
                        target_class=None,
                        intended_class=[np.random.randint(num_classes)])

        tm = self.args.threatmodel
        if tm == 'single-class':
            if fixed_p is not None:
                if fixed_p == target_class:
                    raise ValueError("single-class requires poison_class != target_class.")
                poison_class = fixed_p
                intended_class = [fixed_p] * self.args.targets
            else:
                poison_class = intended_class[0]
        elif tm == 'third-party':
            if fixed_p is not None:
                if fixed_p == target_class:
                    raise ValueError("third-party requires poison_class != target_class.")
                if fixed_p == intended_class[0]:
                    remaining = [c for c in list_intentions if c != fixed_p]
                    intended_class = [np.random.choice(remaining)] * self.args.targets
                poison_class = fixed_p
            else:
                remaining = [c for c in list_intentions if c != intended_class[0]]
                poison_class = np.random.choice(remaining)
        elif tm == 'self-betrayal':
            if fixed_p is not None and fixed_p != target_class:
                raise ValueError("self-betrayal requires poison_class == target_class.")
            poison_class = target_class
        elif tm == 'random-subset':
            if fixed_p is not None:
                raise ValueError("random-subset threatmodel forbids fixing --poison_class.")
            poison_class = None
        elif tm == 'random-subset-random-targets':
            if fixed_t is not None:
                raise ValueError("random-subset-random-targets forbids fixing --target_class.")
            if fixed_p is not None:
                raise ValueError("random-subset-random-targets forbids fixing --poison_class.")
            target_class = None
            intended_class = np.random.randint(num_classes, size=self.args.targets)
            poison_class = None
        else:
            raise NotImplementedError('Unknown threat model.')

        return dict(poison_budget=self.args.budget, target_num=self.args.targets,
                    poison_class=poison_class, target_class=target_class,
                    intended_class=intended_class)

    def _choose_poisons_randomly(self):
        """Copied from KettleRandom._choose_poisons_randomly to keep this mode independent."""
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        if self.poison_setup['target_class'] is not None:
            class_ids = []
            for index in range(len(self.validset)):
                target, idx = self.validset.get_target(index)
                if target == self.poison_setup['target_class']:
                    class_ids.append(idx)
            self.target_ids = np.random.choice(class_ids, size=self.args.targets, replace=False)
        else:
            total_ids = []
            for index in range(len(self.validset)):
                _, idx = self.validset.get_target(index)
                total_ids.append(idx)
            self.target_ids = np.random.choice(total_ids, size=self.args.targets, replace=False)

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        return poisonset, targetset, validset
