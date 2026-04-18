"""Evaluation helpers for artifact-backed dual-attacker jobs."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

import forest
from forest.dual_attack.kettle import ArtifactKettle
from forest.utils import set_random_seed
from forest.victims.victim_base import FINETUNING_LR_DROP


def build_artifact_kettle(args, setup, artifacts, merged_indices):
    victim_probe = forest.Victim(args, setup=setup)
    kettle = ArtifactKettle(
        args,
        victim_probe.defs.batch_size,
        victim_probe.defs.augmentations,
        victim_probe.defs.mixing_method,
        artifacts,
        merged_indices,
        setup=setup,
    )
    return kettle, victim_probe


def _evaluate_valid_accuracy(network, dataloader, setup):
    total = 0
    correct = 0
    network.eval()
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=forest.consts.NON_BLOCKING)
            predictions = torch.argmax(network(inputs), dim=1)
            total += labels.shape[0]
            correct += (predictions == labels).sum().item()
    return 0.0 if total == 0 else correct / total


def _prepare_reference_model(model, args, kettle):
    if args.scenario in ['transfer', 'finetuning'] or args.pretrain_dataset is not None:
        model.train(kettle, max_epoch=args.max_epoch)


@contextmanager
def _victim_seed_modelkey_override(args):
    original_modelkey = args.modelkey
    args.modelkey = None
    try:
        yield
    finally:
        args.modelkey = original_modelkey


def _prepare_model_for_seed(model, args, seed):
    with _victim_seed_modelkey_override(args):
        if args.scenario == 'from-scratch':
            model.initialize(seed=seed)
        elif args.scenario == 'transfer':
            model.load_feature_representation()
            model.reinitialize_last_layer(reduce_lr_factor=1.0, seed=seed)
        elif args.scenario == 'finetuning':
            set_random_seed(seed)
            model.load_feature_representation()
            model.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)
        else:
            raise ValueError(f'Unsupported scenario {args.scenario}.')


def _target_rows(
    *,
    experiment,
    job,
    run_type,
    victim_seed,
    victim_run_id,
    network,
    kettle,
    attack_artifacts,
    clean_accuracy,
    overlap_stats,
):
    class_names = kettle.trainset.classes
    rows = []
    network.eval()
    with torch.no_grad():
        target_images = torch.stack([entry[0] for entry in kettle.targetset]).to(**kettle.setup)
        outputs = network(target_images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    for attack_idx, artifact in enumerate(attack_artifacts):
        adv_class = int(artifact['target_adv_class'])
        true_class = int(artifact['target_true_class'])
        loss = F.cross_entropy(outputs[attack_idx:attack_idx + 1], torch.tensor([adv_class], device=outputs.device))
        rows.append(dict(
            experiment_id=experiment['experiment_id'],
            family=experiment['family'],
            job_id=job['job_id'],
            pairing_id=job.get('pairing_id', ''),
            run_type=run_type,
            victim_run_id=victim_run_id,
            victim_seed=victim_seed,
            attacker_id=artifact['attacker']['attacker_id'],
            source_class=int(artifact['source_class']),
            source_class_name=artifact['source_class_name'],
            target_index=int(artifact['target_index']),
            target_true_class=true_class,
            target_true_class_name=artifact['target_true_class_name'],
            target_adv_class=adv_class,
            target_adv_class_name=artifact['target_adv_class_name'],
            predicted_class=int(predictions[attack_idx].item()),
            predicted_class_name=class_names[int(predictions[attack_idx].item())],
            success=int(predictions[attack_idx].item() == adv_class),
            adv_confidence=float(probabilities[attack_idx, adv_class].item()),
            true_confidence=float(probabilities[attack_idx, true_class].item()),
            target_loss=float(loss.item()),
            clean_accuracy=float(clean_accuracy),
            brew_artifact_path=artifact['artifact_path'],
            brew_job_id=artifact['job_id'],
            brew_loss=artifact.get('brew_loss'),
            source_target_distance=artifact['attacker'].get('source_target_distance'),
            source_target_rank=artifact['attacker'].get('source_target_rank'),
            condition=job.get('condition', artifact['attacker'].get('bucket', '')),
            distance_bucket=job.get('distance_bucket', artifact['attacker'].get('bucket', '')),
            symmetric=job.get('symmetric', ''),
            pair_distance_rank=job.get('pair_distance_rank', ''),
            pair_distance_rank_fraction=job.get('pair_distance_rank_fraction', ''),
            source_source_distance=job.get('source_source_distance', ''),
            source_pair_label=job.get('source_pair_label', ''),
            source_stratum=job.get('source_stratum', ''),
            source_source_rank=job.get('source_source_rank', ''),
            motif_label=job.get('motif_label', ''),
            motif_rank=job.get('motif_rank', ''),
            alignment_type=job.get('alignment_type', ''),
            S=job.get('S', ''),
            T=job.get('T', ''),
            G=job.get('G', ''),
            a_self=job.get('a_self', ''),
            b_self=job.get('b_self', ''),
            a_cross=job.get('a_cross', ''),
            b_cross=job.get('b_cross', ''),
            target_target_distance=job.get('target_target_distance', ''),
            cross_alignment_gap=job.get('cross_alignment_gap', ''),
            overlap_policy=job.get('overlap_policy', ''),
            overlap_seed=job.get('overlap_seed', ''),
            overlap_total=overlap_stats.get('overlap_total', 0),
            overlaps_lost_by_attacker=overlap_stats.get('lost_by_attacker', {}).get(
                artifact['attacker']['attacker_id'], 0
            ),
        ))
    return rows


def evaluate_job(args, experiment, job, artifacts, poison_delta, merged_indices, overlap_stats, run_type, setup):
    kettle, model = build_artifact_kettle(args, setup, artifacts, merged_indices)
    _prepare_reference_model(model, args, kettle)

    rows = []
    for victim_run_id, victim_seed in enumerate(job['victim_seeds']):
        _prepare_model_for_seed(model, args, victim_seed)
        stats = model._iterate(kettle, poison_delta=poison_delta)
        clean_accuracy = stats['valid_accs'][-1] if len(stats.get('valid_accs', [])) > 0 else _evaluate_valid_accuracy(
            model.model, kettle.validloader, kettle.setup
        )
        rows.extend(_target_rows(
            experiment=experiment,
            job=job,
            run_type=run_type,
            victim_seed=victim_seed,
            victim_run_id=victim_run_id,
            network=model.model,
            kettle=kettle,
            attack_artifacts=artifacts,
            clean_accuracy=clean_accuracy,
            overlap_stats=overlap_stats,
        ))
    return rows
