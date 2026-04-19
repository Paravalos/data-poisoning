"""Human-friendly rendering helpers for dual-attacker experiment specs."""

from __future__ import annotations


def _class_name(experiment, class_idx):
    return experiment['class_names'][int(class_idx)]


def summarize_experiment(experiment):
    metadata = experiment.get('metadata', {})
    lines = [f'Experiment: {experiment["experiment_id"]}', f'Family: {experiment["family"]}']

    if 'shared_target_class' in metadata:
        lines.extend([
            f'Target class: {_class_name(experiment, metadata["shared_target_class"])} ({metadata["shared_target_class"]})',
            (
                'Fixed attacker A source: '
                f'{_class_name(experiment, metadata["fixed_attacker_a_source_class"])} '
                f'({metadata["fixed_attacker_a_source_class"]})'
            ),
        ])
    elif 'fixed_attacker_a_source_class' in metadata:
        lines.append(
            'Fixed attacker A source: '
            f'{_class_name(experiment, metadata["fixed_attacker_a_source_class"])} '
            f'({metadata["fixed_attacker_a_source_class"]})'
        )

    if 'planning_seed' in metadata:
        lines.append(f'Planning seed: {metadata["planning_seed"]}')

    lines.extend([
        f'Repeats: {metadata["repeats"]}',
        f'Victim seeds: {", ".join(str(seed) for seed in metadata["victim_seeds"])}',
        f'Brew jobs: {len(experiment.get("brew_jobs", []))}',
        f'Solo jobs: {len(experiment.get("solo_jobs", []))}',
        f'Dual jobs: {len(experiment.get("dual_jobs", []))}',
        'Dual pairings:',
    ])

    for dual_job in experiment.get('dual_jobs', []):
        attackers = dual_job['attackers']
        left = attackers[0]
        right = attackers[1]
        lines.append(
            '  - '
            f'{dual_job["job_id"]}: '
            f'{_class_name(experiment, left["source_class"])}[{left["target_index"]}]'
            f'->{_class_name(experiment, left["target_adv_class"])} '
            f'vs {_class_name(experiment, right["source_class"])}[{right["target_index"]}]'
            f'->{_class_name(experiment, right["target_adv_class"])}'
        )

    if metadata.get('sampled_target_indices'):
        lines.append('Sampled target indices:')
        for class_idx, indices in sorted(metadata.get('sampled_target_indices', {}).items(), key=lambda item: int(item[0])):
            lines.append(f'  - {_class_name(experiment, class_idx)} ({class_idx}): {indices}')

    return '\n'.join(lines)
