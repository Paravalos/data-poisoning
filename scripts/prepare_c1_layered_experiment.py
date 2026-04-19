"""Prepare a layered C1 dual-attacker experiment: 2 targets x 3 anchors x 6 partners.

Each dual job pairs an anchor attacker A with a partner attacker B, both targeting the
same shared class T (the "shared target" case of C1). Anchors and partners are chosen
to span d(A, T) and d(A, B) respectively, using the precomputed class-distance artifact.
"""

from __future__ import annotations

import argparse

from _dual_attack_script_utils import ensure_repo_root

ensure_repo_root(__file__)

from forest.dual_attack.prepare import (
    add_shared_prepare_arguments,
    build_experiment_plan,
    compile_experiment_plan,
    load_preparation_context,
    save_prepared_experiment_outputs,
)
from forest.dual_attack.planners import (
    _build_attacker_spec,
    _class_index,
    _source_target_ranks,
)


# 2 targets x 3 anchors x 6 partners. Anchors span d(A, T), partners span d(A, B).
# Class selections chosen from the CIFAR-10 ResNet18 class-mean cosine distance matrix
# to maximise distance spread while keeping the design balanced.
LAYERED_DESIGN = [
    dict(
        target='airplane',
        anchors=[
            dict(name='bird', role='near',
                 partners=['deer', 'cat', 'dog', 'horse', 'ship', 'automobile']),
            dict(name='cat', role='mid',
                 partners=['dog', 'bird', 'deer', 'horse', 'ship', 'truck']),
            dict(name='frog', role='far',
                 partners=['cat', 'bird', 'deer', 'dog', 'truck', 'horse']),
        ],
    ),
    dict(
        target='dog',
        anchors=[
            dict(name='cat', role='near',
                 partners=['bird', 'deer', 'frog', 'airplane', 'horse', 'ship']),
            dict(name='horse', role='mid',
                 partners=['deer', 'bird', 'cat', 'airplane', 'truck', 'frog']),
            dict(name='ship', role='far',
                 partners=['airplane', 'truck', 'automobile', 'cat', 'bird', 'horse']),
        ],
    ),
]

# Force eps=8 for this experiment family (overrides the shared eps=16 default).
EPS_OVERRIDE = 8


def _validate_design(design, class_names):
    for target_entry in design:
        target = target_entry['target']
        if target not in class_names:
            raise ValueError(f'Unknown target class {target}.')
        for anchor_entry in target_entry['anchors']:
            anchor = anchor_entry['name']
            if anchor == target:
                raise ValueError(f'Anchor {anchor} cannot equal target {target}.')
            if anchor not in class_names:
                raise ValueError(f'Unknown anchor class {anchor}.')
            for partner in anchor_entry['partners']:
                if partner == anchor or partner == target:
                    raise ValueError(
                        f'Partner {partner} must differ from anchor {anchor} and target {target}.'
                    )
                if partner not in class_names:
                    raise ValueError(f'Unknown partner class {partner}.')


def _pairing_id(target_name, anchor_name, partner_name, repeat_slot):
    return f'{target_name}__{anchor_name}_vs_{partner_name}_rep{repeat_slot + 1}'


def build_c1_layered_plan(*, preparation_context, design):
    class_names = preparation_context.class_names
    distance_matrix = preparation_context.distance_matrix
    repeats = preparation_context.repeats
    overlap_seed_base = preparation_context.planner_defaults['overlap_seed_base']

    _validate_design(design, class_names)

    dual_requests = []
    target_summaries = []

    for target_entry in design:
        target_name = target_entry['target']
        target_idx = _class_index(class_names, target_name)
        source_target_ranks = _source_target_ranks(target_name, preparation_context.rankings)

        anchor_summaries = []
        for anchor_entry in target_entry['anchors']:
            anchor_name = anchor_entry['name']
            anchor_idx = _class_index(class_names, anchor_name)
            anchor_role = anchor_entry['role']

            partner_summaries = []
            for partner_name in anchor_entry['partners']:
                partner_idx = _class_index(class_names, partner_name)
                source_source_distance = float(distance_matrix[anchor_idx, partner_idx])

                for repeat_slot in range(repeats):
                    attacker_requests = [
                        dict(
                            source_class_idx=anchor_idx,
                            target_class_idx=target_idx,
                            repeat_slot=repeat_slot,
                            source_target_distance=float(distance_matrix[target_idx, anchor_idx]),
                            source_target_rank=source_target_ranks[anchor_idx],
                        ),
                        dict(
                            source_class_idx=partner_idx,
                            target_class_idx=target_idx,
                            repeat_slot=repeat_slot,
                            source_target_distance=float(distance_matrix[target_idx, partner_idx]),
                            source_target_rank=source_target_ranks[partner_idx],
                        ),
                    ]
                    pairing_id = _pairing_id(target_name, anchor_name, partner_name, repeat_slot)
                    dual_requests.append(dict(
                        pairing_id=pairing_id,
                        attacker_requests=attacker_requests,
                        overlap_seed=overlap_seed_base + repeat_slot,
                        metadata=dict(
                            condition=f'{target_name}__{anchor_name}_vs_{partner_name}',
                            distance_bucket='layered_anchor_partner_sweep',
                            shared_target_class=target_idx,
                            shared_target_class_name=target_name,
                            anchor_source_class=anchor_idx,
                            anchor_source_class_name=anchor_name,
                            anchor_role=anchor_role,
                            partner_source_class=partner_idx,
                            partner_source_class_name=partner_name,
                            anchor_target_distance=float(distance_matrix[target_idx, anchor_idx]),
                            partner_target_distance=float(distance_matrix[target_idx, partner_idx]),
                            source_source_distance=source_source_distance,
                        ),
                    ))

                partner_summaries.append(dict(
                    name=partner_name,
                    anchor_partner_distance=source_source_distance,
                    partner_target_distance=float(distance_matrix[target_idx, partner_idx]),
                ))

            anchor_summaries.append(dict(
                name=anchor_name,
                role=anchor_role,
                anchor_target_distance=float(distance_matrix[target_idx, anchor_idx]),
                partners=partner_summaries,
            ))

        target_summaries.append(dict(name=target_name, anchors=anchor_summaries))

    return build_experiment_plan(
        family='C1',
        experiment_metadata=dict(
            repeats=repeats,
            victim_seeds=list(preparation_context.planner_defaults['victim_seeds']),
            distance_artifact_path=preparation_context.distance_artifact_path,
            pair_selection_strategy='layered_target_anchor_partner_sweep',
            layered_design=target_summaries,
            dual_job_count=len(dual_requests),
        ),
        dual_requests=dual_requests,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_prepare_arguments(
        parser,
        fixed_attacker_help='Unused in the layered C1 design (anchors are enumerated from LAYERED_DESIGN).',
    )
    parser.set_defaults(repeats=8)
    args = parser.parse_args()

    preparation_context = load_preparation_context(args)
    # Force eps=8 for this experiment family.
    preparation_context.common_args['eps'] = EPS_OVERRIDE

    experiment_plan = build_c1_layered_plan(
        preparation_context=preparation_context,
        design=LAYERED_DESIGN,
    )
    experiment = compile_experiment_plan(
        preparation_context=preparation_context,
        experiment_plan=experiment_plan,
        build_attacker_spec=_build_attacker_spec,
    )
    save_prepared_experiment_outputs(
        preparation_context=preparation_context,
        experiment=experiment,
        experiment_plan=experiment_plan,
        save_plan_yaml_enabled=args.save_plan_yaml,
    )
    print(
        f'Saved layered C1 experiment spec to {preparation_context.experiment_path} '
        f'(duals={len(experiment["dual_jobs"])}, solos={len(experiment["solo_jobs"])}, '
        f'brews={len(experiment["brew_jobs"])}, eps={preparation_context.common_args["eps"]}).'
    )
