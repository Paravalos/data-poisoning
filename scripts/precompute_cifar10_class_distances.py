"""Precompute a CIFAR-10 class distance matrix from clean ResNet18 penultimate features."""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _load_options():
    spec = importlib.util.spec_from_file_location('forest_options', os.path.join(REPO_ROOT, 'forest', 'options.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.options


def _feature_extractor(model):
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    headless_model = torch.nn.Sequential(*list(target_model.children())[:-1], torch.nn.Flatten())
    headless_model.to(next(target_model.parameters()).device)
    headless_model.eval()
    return headless_model


def _compute_class_centroids(feature_model, trainset, batch_size, setup):
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    class_sums = None
    class_counts = torch.zeros(len(trainset.classes), dtype=torch.long)

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(**setup)
            features = feature_model(images).detach().cpu()

            if class_sums is None:
                class_sums = torch.zeros(len(trainset.classes), features.shape[1], dtype=features.dtype)

            for class_idx in range(len(trainset.classes)):
                class_mask = labels == class_idx
                if class_mask.any():
                    class_sums[class_idx] += features[class_mask].sum(dim=0)
                    class_counts[class_idx] += class_mask.sum()

    if class_sums is None:
        raise ValueError('No features were extracted from the CIFAR-10 training set.')
    if torch.any(class_counts == 0):
        raise ValueError('At least one CIFAR-10 class has zero samples, cannot compute centroids.')

    return class_sums / class_counts.unsqueeze(1)


def _compute_rankings(distance_matrix, class_names):
    rankings = {}
    for target_idx, class_name in enumerate(class_names):
        ranked_indices = torch.argsort(distance_matrix[target_idx]).tolist()
        rankings[class_name] = [
            dict(class_index=source_idx,
                 class_name=class_names[source_idx],
                 cosine_distance=float(distance_matrix[target_idx, source_idx]))
            for source_idx in ranked_indices if source_idx != target_idx
        ]
    return rankings


if __name__ == "__main__":
    parser = _load_options()()
    parser.add_argument('--output', default=None, type=str, help='Path to save the class distance matrix artifact.')
    args = parser.parse_args()

    import forest

    torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
    torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)
    if args.deterministic:
        forest.utils.set_deterministic()

    args.dataset = 'CIFAR10'
    args.net = ['ResNet18']
    args.modelkey = 0
    args.ensemble = 1
    args.pretrained_model = False
    args.skip_clean_training = False
    args.vruns = 0

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)

    model.train(data, max_epoch=args.max_epoch)
    model.model.eval()

    feature_model = _feature_extractor(model.model)
    class_centroids = _compute_class_centroids(feature_model, data.trainset, model.defs.batch_size, setup)
    normalized_centroids = F.normalize(class_centroids, dim=1)
    distance_matrix = 1 - normalized_centroids @ normalized_centroids.t()
    distance_matrix.fill_diagonal_(0.0)

    class_names = list(data.trainset.classes)
    rankings = _compute_rankings(distance_matrix, class_names)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.expanduser(args.modelsave_path),
                                   'class_distance_matrix_cifar10_resnet18_modelkey0.pt')
    else:
        output_path = os.path.expanduser(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    artifact = dict(
        distance_matrix=distance_matrix,
        class_centroids=class_centroids,
        class_names=class_names,
        rankings=rankings,
        modelkey=args.modelkey,
        model_init_seed=model.model_init_seed,
        dataset=args.dataset,
        net=args.net[0],
        modelsave_path=os.path.expanduser(args.modelsave_path),
        clean_model_cache_path=model._compute_clean_model_cache_path() if hasattr(model, '_compute_clean_model_cache_path') else None,
        optimization=args.optimization,
        epochs=model.defs.epochs if args.epochs is None else args.epochs,
        batch_size=model.defs.batch_size,
        data_path=os.path.expanduser(args.data_path),
    )
    torch.save(artifact, output_path)

    print(f'Saved class distance matrix to {output_path}.')
