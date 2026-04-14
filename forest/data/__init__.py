"""Basic data handling."""
import torch

from .kettle_random_experiment import KettleRandom
from .kettle_det_experiment import KettleDeterministic
from .kettle_benchmark_experiment import KettleBenchmark
from .kettle_fixed_class_experiment import KettleFixedClass
from .kettle_external import KettleExternal

__all__ = ['Kettle', 'KettleExternal']


def Kettle(args, batch_size, augmentations, mixing_method=dict(type=None, strength=0.0),
           setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Interface to connect to a kettle [data] child class."""

    fixed_class_requested = (
        getattr(args, 'target_class', None) is not None
        or getattr(args, 'poison_class', None) is not None
    )

    if args.poisonkey is None:
        if args.benchmark != '':
            return KettleBenchmark(args, batch_size, augmentations, mixing_method, setup)
        if fixed_class_requested:
            return KettleFixedClass(args, batch_size, augmentations, mixing_method, setup)
        return KettleRandom(args, batch_size, augmentations, mixing_method, setup)

    if '-' in args.poisonkey:
        # Fully deterministic triplet path; fixed-class flags are ignored here
        # because deterministic mode already pins both classes.
        return KettleDeterministic(args, batch_size, augmentations, mixing_method, setup)
    if fixed_class_requested:
        return KettleFixedClass(args, batch_size, augmentations, mixing_method, setup)
    return KettleRandom(args, batch_size, augmentations, mixing_method, setup)
