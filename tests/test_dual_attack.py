import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from forest.data.kettle_det_experiment import select_deterministic_poison_ids
from forest.dual_attack.artifacts import load_brew_artifact, save_brew_artifact
from forest.dual_attack.config import c1_planner_defaults, common_args_defaults, load_defaults, planner_defaults, scheduler_defaults
from forest.dual_attack.eval import _prepare_model_for_seed, _target_rows
from forest.dual_attack.experiment import build_args_namespace, load_experiment, save_experiment
from forest.dual_attack.planners import build_c5_experiment
from forest.dual_attack.prepare import (
    add_shared_prepare_arguments,
    assign_target_indices,
    build_dual_job,
    ensure_registered_attacker,
    materialize_attackers,
    save_plan_yaml,
)
from forest.dual_attack.runtime import _artifact_poison_indices, _brew_cache_path, _try_reuse_brew_artifact, resolve_dual_overlap, validate_brew_artifact
from forest.dual_attack.summary import summarize_experiment
from forest.dual_attack.submitter import build_sbatch_command, plan_submission_specs


def _load_script_module(module_name):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_root = repo_root / 'scripts'
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    module_path = repo_root / 'scripts' / f'{module_name}.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_c1_experiment = _load_script_module('prepare_c1_experiment').build_c1_experiment
build_c2_experiment = _load_script_module('prepare_c2_experiment').build_c2_experiment
build_c1_plan = _load_script_module('prepare_c1_experiment').build_c1_plan
build_c2_plan = _load_script_module('prepare_c2_experiment').build_c2_plan


class DualAttackPrepareConfigTests(unittest.TestCase):
    def test_add_shared_prepare_arguments_only_exposes_planner_flags(self):
        parser = argparse.ArgumentParser()
        add_shared_prepare_arguments(parser, fixed_attacker_help='demo')
        args = parser.parse_args(['--distance_artifact', 'artifact.pt', '--output', 'experiment.json'])

        self.assertTrue(hasattr(args, 'fixed_attacker_a_source_class'))
        self.assertTrue(hasattr(args, 'repeats'))
        self.assertTrue(hasattr(args, 'save_plan_yaml'))
        self.assertFalse(hasattr(args, 'planning_seed'))
        self.assertFalse(hasattr(args, 'victim_seeds'))
        self.assertFalse(hasattr(args, 'slurm_gpu'))

    def test_common_args_defaults_include_shared_runtime_settings(self):
        defaults = common_args_defaults()
        self.assertEqual(defaults['dataset'], 'CIFAR10')
        self.assertEqual(defaults['data_path'], '~/data')
        self.assertEqual(defaults['modelkey'], 0)
        self.assertTrue(defaults['cache_clean_model'])
        self.assertTrue(defaults['randomize_deterministic_poison_ids'])

    def test_planner_and_scheduler_defaults_keep_non_cli_settings_centralized(self):
        self.assertEqual(planner_defaults()['victim_seeds'], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(planner_defaults()['overlap_seed_base'], 0)
        self.assertEqual(scheduler_defaults()['gpu'], 'l40s')
        self.assertEqual(scheduler_defaults()['mem'], '20G')

    def test_yaml_defaults_are_loaded_from_central_file(self):
        defaults = load_defaults()
        self.assertIn('common_args', defaults)
        self.assertIn('planner_defaults', defaults)
        self.assertEqual(c1_planner_defaults()['shared_target_class'], 'airplane')

    def test_ensure_registered_attacker_registers_once_per_key(self):
        build_calls = []
        attacker_specs = {}
        artifact_by_attacker = {}
        brew_jobs = []
        solo_jobs = []

        def build_attacker_spec(**kwargs):
            build_calls.append(kwargs)
            return dict(
                job_id='brew_demo',
                attacker=dict(attacker_id='demo', brew_job_id='brew_demo'),
                arg_overrides=dict(name='brew_demo'),
                artifact_path='/tmp/brew_demo.pt',
            )

        first = ensure_registered_attacker(
            attacker_key=('demo',),
            build_attacker_spec=build_attacker_spec,
            attacker_specs=attacker_specs,
            artifact_by_attacker=artifact_by_attacker,
            brew_jobs=brew_jobs,
            solo_jobs=solo_jobs,
            solo_dir='/tmp',
            victim_seeds=[0],
        )
        second = ensure_registered_attacker(
            attacker_key=('demo',),
            build_attacker_spec=build_attacker_spec,
            attacker_specs=attacker_specs,
            artifact_by_attacker=artifact_by_attacker,
            brew_jobs=brew_jobs,
            solo_jobs=solo_jobs,
            solo_dir='/tmp',
            victim_seeds=[0],
        )

        self.assertEqual(first, second)
        self.assertEqual(len(build_calls), 1)
        self.assertEqual(len(brew_jobs), 1)
        self.assertEqual(len(solo_jobs), 1)

    def test_materialize_attackers_and_build_dual_job_share_common_wiring(self):
        attacker_specs = {}
        artifact_by_attacker = {}
        brew_jobs = []
        solo_jobs = []

        def build_attacker_spec(**kwargs):
            attacker_id = kwargs['attacker_key_name']
            return dict(
                job_id=f'brew_{attacker_id}',
                attacker=dict(attacker_id=attacker_id, brew_job_id=f'brew_{attacker_id}'),
                arg_overrides=dict(name=f'brew_{attacker_id}'),
                artifact_path=f'/tmp/brew_{attacker_id}.pt',
            )

        attackers = materialize_attackers(
            attacker_requests=[
                dict(attacker_key=('a',), attacker_key_name='a'),
                dict(attacker_key=('b',), attacker_key_name='b'),
            ],
            build_attacker_spec=build_attacker_spec,
            attacker_specs=attacker_specs,
            artifact_by_attacker=artifact_by_attacker,
            brew_jobs=brew_jobs,
            solo_jobs=solo_jobs,
            solo_dir='/tmp',
            victim_seeds=[0, 1],
        )
        dual_job = build_dual_job(
            pairing_id='pair1',
            attackers=attackers,
            artifact_by_attacker=artifact_by_attacker,
            victim_seeds=[0, 1],
            overlap_seed=7,
            dual_dir='/tmp',
            extra_fields=dict(condition='demo'),
        )

        self.assertEqual([attacker['attacker_id'] for attacker in attackers], ['a', 'b'])
        self.assertEqual(len(brew_jobs), 2)
        self.assertEqual(len(solo_jobs), 2)
        self.assertEqual(dual_job['job_id'], 'dual_pair1')
        self.assertEqual(dual_job['brew_artifact_paths'], ['/tmp/brew_a.pt', '/tmp/brew_b.pt'])
        self.assertEqual(dual_job['condition'], 'demo')

    def test_assign_target_indices_uses_source_target_and_repeat(self):
        first = assign_target_indices(
            attacker_requests=[dict(source_class_idx=5, target_class_idx=0, repeat_slot=0)],
            class_to_valid_indices={5: [500, 501, 502, 503]},
        )[0]
        second = assign_target_indices(
            attacker_requests=[dict(source_class_idx=5, target_class_idx=0, repeat_slot=1)],
            class_to_valid_indices={5: [500, 501, 502, 503]},
        )[0]

        self.assertNotEqual(first['target_index'], second['target_index'])
        self.assertEqual(first['attacker_key'][:3], (5, 0, 0))
        self.assertEqual(second['attacker_key'][:3], (5, 1, 0))

    def test_assign_target_indices_is_order_independent_for_distinct_attackers(self):
        class_to_valid_indices = {5: [500, 501, 502, 503]}
        first = assign_target_indices(
            attacker_requests=[
                dict(source_class_idx=5, target_class_idx=0, repeat_slot=0),
                dict(source_class_idx=5, target_class_idx=3, repeat_slot=0),
            ],
            class_to_valid_indices=class_to_valid_indices,
        )
        second = assign_target_indices(
            attacker_requests=[
                dict(source_class_idx=5, target_class_idx=3, repeat_slot=0),
                dict(source_class_idx=5, target_class_idx=0, repeat_slot=0),
            ],
            class_to_valid_indices=class_to_valid_indices,
        )

        first_by_target = {request['target_class_idx']: request['target_index'] for request in first}
        second_by_target = {request['target_class_idx']: request['target_index'] for request in second}
        self.assertEqual(first_by_target, second_by_target)

    def test_assign_target_indices_rejects_duplicate_attacker_identity(self):
        with self.assertRaises(ValueError):
            assign_target_indices(
                attacker_requests=[
                    dict(source_class_idx=5, target_class_idx=0, repeat_slot=0),
                    dict(source_class_idx=5, target_class_idx=0, repeat_slot=0),
                ],
                class_to_valid_indices={5: [500, 501, 502, 503]},
            )

    def test_save_plan_yaml_writes_readable_yaml(self):
        plan = dict(
            family='C1',
            experiment_metadata=dict(repeats=1),
            dual_requests=[
                dict(
                    pairing_id='pair1',
                    overlap_seed=7,
                    attacker_requests=[
                        dict(source_class_idx=5, target_class_idx=0, repeat_slot=0),
                        dict(source_class_idx=3, target_class_idx=0, repeat_slot=0),
                    ],
                    metadata=dict(condition='demo'),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'plan.yaml')
            save_plan_yaml(plan, ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog'], path)
            with open(path, 'r') as handle:
                payload = handle.read()
        self.assertIn('family: C1', payload)
        self.assertIn('source_class_name: dog', payload)
        self.assertIn('target_class_name: airplane', payload)


class DualAttackPlannerTests(unittest.TestCase):
    def setUp(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.distance_matrix = torch.zeros((10, 10), dtype=torch.float)
        pair_distances = {
            (0, 1): 0.6062602996826172,
            (0, 2): 0.49285149574279785,
            (0, 3): 0.5680851936340332,
            (0, 4): 0.6046143770217896,
            (0, 5): 0.6689342260360718,
            (0, 6): 0.681496262550354,
            (0, 7): 0.6319902539253235,
            (0, 8): 0.507644772529602,
            (0, 9): 0.5797238945960999,
            (1, 2): 0.6768210000000000,
            (1, 3): 0.6126067638397217,
            (1, 4): 0.6950000000000000,
            (1, 5): 0.6543991565704346,
            (1, 6): 0.6590000000000000,
            (1, 7): 0.6492404937744141,
            (1, 8): 0.5940215587615967,
            (1, 9): 0.49285149574279785,
            (2, 3): 0.4896228313446045,
            (2, 4): 0.4731823205947876,
            (2, 5): 0.5690615177154541,
            (2, 6): 0.5400000000000000,
            (2, 7): 0.5737546682357788,
            (2, 8): 0.6460000000000000,
            (2, 9): 0.6590000000000000,
            (3, 4): 0.5025486350059509,
            (3, 5): 0.39107412099838257,
            (3, 6): 0.5170000000000000,
            (3, 7): 0.5820205807685852,
            (3, 8): 0.6090000000000000,
            (3, 9): 0.6187872886657715,
            (4, 5): 0.5543463230133057,
            (4, 6): 0.5835475325584412,
            (4, 7): 0.5446214675903320,
            (4, 8): 0.6710000000000000,
            (4, 9): 0.6820000000000000,
            (5, 6): 0.6073779463768005,
            (5, 7): 0.5162463784217834,
            (5, 8): 0.6781786084175110,
            (5, 9): 0.6590000000000000,
            (6, 7): 0.6753582954406738,
            (6, 8): 0.6640000000000000,
            (6, 9): 0.6480000000000000,
            (7, 8): 0.6868035197257996,
            (7, 9): 0.6561658382415771,
            (8, 9): 0.5802785754203796,
        }
        for (row, col), value in pair_distances.items():
            self.distance_matrix[row, col] = value
            self.distance_matrix[col, row] = value
        self.rankings = {}
        for target_class_idx, target_class_name in enumerate(self.class_names):
            ranked_entries = [
                dict(
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    cosine_distance=float(self.distance_matrix[target_class_idx, class_idx]),
                )
                for class_idx in range(len(self.class_names))
                if class_idx != target_class_idx
            ]
            ranked_entries.sort(key=lambda entry: (entry['cosine_distance'], entry['class_index']))
            self.rankings[target_class_name] = ranked_entries
        self.class_to_valid_indices = {class_idx: [class_idx * 100 + offset for offset in range(8)] for class_idx in range(10)}
        self.common_args = dict(
            dataset='CIFAR10',
            net=['ResNet18'],
            scenario='from-scratch',
            randomize_deterministic_poison_ids=True,
        )

    def test_build_c1_experiment_generates_expected_jobs(self):
        experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=3,
            victim_seeds=[0, 1],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=7,
            distance_artifact_path='artifact.pt',
        )
        self.assertEqual(len(experiment['dual_jobs']), 8 * 3)
        self.assertEqual(len(experiment['solo_jobs']), len(experiment['brew_jobs']))
        self.assertEqual(experiment['class_names'][0], 'airplane')
        self.assertEqual(experiment['metadata']['pair_selection_strategy'], 'fixed_attacker_a_source_sweep')
        self.assertNotIn('planning_seed', experiment['metadata'])
        self.assertTrue(experiment['common_args']['randomize_deterministic_poison_ids'])
        self.assertEqual(experiment['dual_jobs'][0]['attackers'][0]['source_class'], 5)
        self.assertEqual(experiment['dual_jobs'][0]['attackers'][1]['source_class'], 1)
        first_left = experiment['dual_jobs'][0]['attackers'][0]
        first_right = experiment['dual_jobs'][0]['attackers'][1]
        self.assertEqual(first_left['poisonkey'], f'5-0-{first_left["target_index"]}')
        self.assertEqual(first_right['poisonkey'], f'1-0-{first_right["target_index"]}')
        self.assertEqual(first_left['repeat_slot'], 0)
        self.assertEqual(first_right['repeat_slot'], 0)
        self.assertEqual(first_left['poison_ids_seed'], 'src5_rep0')
        self.assertEqual(first_right['poison_ids_seed'], 'src1_rep0')
        self.assertTrue(all(job['attackers'][0]['source_class'] != job['attackers'][1]['source_class']
                            for job in experiment['dual_jobs']))
        self.assertEqual(experiment['dual_jobs'][-1]['attackers'][0]['source_class'], 5)
        self.assertEqual(experiment['dual_jobs'][-1]['attackers'][1]['source_class'], 9)

    def test_save_and_load_experiment_round_trip(self):
        experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=1,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_path = os.path.join(tmpdir, 'experiment.json')
            save_experiment(experiment, experiment_path)
            loaded = load_experiment(experiment_path)
        self.assertEqual(loaded['experiment_id'], 'C1_airplane_v1')
        self.assertEqual(len(loaded['dual_jobs']), 8)
        self.assertNotIn('sampled_target_indices', loaded['metadata'])

    def test_summary_uses_top_level_class_names(self):
        experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=1,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        summary = summarize_experiment(experiment)
        self.assertIn('Target class: airplane (0)', summary)
        self.assertIn('Fixed attacker A source: dog (5)', summary)

    def test_build_c2_experiment_generates_expected_jobs(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            repeats=3,
            victim_seeds=[0, 1],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=7,
            distance_artifact_path='artifact.pt',
        )
        self.assertEqual(experiment['family'], 'C2')
        self.assertEqual(len(experiment['dual_jobs']), 3 * 4 * 3)
        self.assertEqual(len(experiment['solo_jobs']), len(experiment['brew_jobs']))
        self.assertEqual(
            [entry['source_stratum'] for entry in experiment['metadata']['source_strata']],
            ['near', 'medium', 'far'],
        )
        near_pair = experiment['metadata']['source_strata'][0]
        self.assertEqual(near_pair['partner_source_class_name'], 'cat')
        self.assertEqual(near_pair['source_source_rank'], 1)
        self.assertEqual(experiment['dual_jobs'][0]['source_stratum'], 'near')
        self.assertIn(experiment['dual_jobs'][0]['motif_label'], set(experiment['metadata']['motif_labels']))
        self.assertIn('G', experiment['dual_jobs'][0])
        self.assertIn('a_self', experiment['dual_jobs'][0])

    def test_build_c2_experiment_reuses_repeat_slots_across_targets(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            repeats=2,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        dog_attackers = [
            job['attacker']
            for job in experiment['brew_jobs']
            if job['attacker']['source_class'] == self.class_names.index('dog') and job['attacker']['repeat_slot'] == 0
        ]
        self.assertGreater(len(dog_attackers), 1)
        targets_by_adv_class = {}
        for attacker in dog_attackers:
            key = attacker['target_adv_class']
            targets_by_adv_class.setdefault(key, set()).add(attacker['target_index'])
        self.assertTrue(targets_by_adv_class)
        self.assertTrue(all(len(indices) == 1 for indices in targets_by_adv_class.values()))
        self.assertEqual({attacker['poison_ids_seed'] for attacker in dog_attackers}, {'src5_rep0'})

    def test_build_c2_experiment_reuses_same_seed_for_same_attacker_across_pairings(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            repeats=1,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        seed_by_attacker = {}
        for job in experiment['dual_jobs']:
            for attacker in job['attackers']:
                key = (attacker['source_class'], attacker['repeat_slot'])
                seed_by_attacker.setdefault(key, set()).add(attacker['poison_ids_seed'])
        self.assertTrue(seed_by_attacker)
        for seeds in seed_by_attacker.values():
            self.assertEqual(len(seeds), 1)

    def test_c1_and_c2_share_target_and_poison_seed_for_common_attackers(self):
        c1_experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=2,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        c2_experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            repeats=2,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )

        def attacker_map(experiment):
            return {
                (
                    attacker['source_class'],
                    attacker['target_adv_class'],
                    attacker['repeat_slot'],
                ): attacker
                for attacker in (job['attacker'] for job in experiment['brew_jobs'])
            }

        c1_by_key = attacker_map(c1_experiment)
        c2_by_key = attacker_map(c2_experiment)
        common_keys = sorted(set(c1_by_key) & set(c2_by_key))
        self.assertTrue(common_keys)
        for key in common_keys:
            self.assertEqual(c1_by_key[key]['target_index'], c2_by_key[key]['target_index'])
            self.assertEqual(c1_by_key[key]['poison_ids_seed'], c2_by_key[key]['poison_ids_seed'])

    def test_c2_summary_uses_top_level_class_names(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            repeats=1,
            victim_seeds=[0],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        summary = summarize_experiment(experiment)
        self.assertIn('Family: C2', summary)
        self.assertIn('Fixed attacker A source: dog (5)', summary)

    def test_build_c5_experiment_derives_targets_from_c1_pairing(self):
        c1_experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=3,
            victim_seeds=[0, 1],
            common_args=self.common_args,
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=7,
            distance_artifact_path='artifact.pt',
        )
        class_to_train_indices = {class_idx: [class_idx * 10_000 + offset for offset in range(200)] for class_idx in range(10)}

        c5_experiment = build_c5_experiment(
            experiment_id='C5_airplane_overlap_v1',
            c1_experiment=c1_experiment,
            class_to_train_indices=class_to_train_indices,
            pair_condition='fixed_dog_vs_cat',
            overlap_percentages=[0, 25, 50, 100],
            output_root='artifacts/c5',
            scheduler={},
            source_experiment_path='artifacts/c1/c1_experiment.json',
        )

        self.assertEqual(c5_experiment['family'], 'C5')
        self.assertEqual(len(c5_experiment['dual_jobs']), 3 * 4)
        self.assertEqual(c5_experiment['metadata']['source_experiment_id'], 'C1_airplane_v1')
        self.assertEqual(c5_experiment['metadata']['pair_condition'], 'fixed_dog_vs_cat')
        self.assertEqual(c5_experiment['metadata']['merge_rule'], 'sum_clipped')

        c1_dog_cat_jobs = [job for job in c1_experiment['dual_jobs'] if job['condition'] == 'fixed_dog_vs_cat']
        c1_job_by_pairing = {job['pairing_id']: job for job in c1_dog_cat_jobs}
        c5_rep1_jobs = [job for job in c5_experiment['dual_jobs'] if job['pairing_id'].endswith('rep1')]
        c1_rep1 = sorted(c1_dog_cat_jobs, key=lambda job: job['pairing_id'])[0]
        c5_rep1_targets = {
            overlap_job['overlap_percentage']: [attacker['target_index'] for attacker in overlap_job['attackers']]
            for overlap_job in c5_rep1_jobs
        }
        for overlap_percentage in [0, 25, 50, 100]:
            self.assertEqual(
                c5_rep1_targets[overlap_percentage],
                [attacker['target_index'] for attacker in c1_rep1['attackers']],
            )

        poison_budget_count = c5_experiment['metadata']['poison_budget_count']
        for job in c5_experiment['dual_jobs']:
            self.assertEqual(job['merge_rule'], 'sum_clipped')
            self.assertEqual(job['overlap_policy'], 'explicit_shared_ids')
            self.assertEqual(job['condition'], 'fixed_dog_vs_cat')
            shared_count = len(set(job['attackers'][0]['explicit_poison_ids']) & set(job['attackers'][1]['explicit_poison_ids']))
            self.assertEqual(shared_count, int(round(poison_budget_count * job['overlap_fraction'])))
            source_job = c1_job_by_pairing[job['c1_pairing_id']]
            for attacker in job['attackers']:
                self.assertEqual(len(attacker['explicit_poison_ids']), poison_budget_count)
                self.assertIn(attacker['c1_attacker_id'], {item['attacker_id'] for item in source_job['attackers']})


class DualAttackRuntimeTests(unittest.TestCase):
    def test_build_args_namespace_preserves_lists(self):
        args = build_args_namespace(
            common_args=dict(dataset='CIFAR10', net=['ResNet18'], scenario='from-scratch'),
            arg_overrides=dict(name='demo', poisonkey='1-0-42'),
        )
        self.assertEqual(args.net, ['ResNet18'])
        self.assertEqual(args.poisonkey, '1-0-42')

    def test_resolve_dual_overlap_is_deterministic(self):
        left = dict(
            poison_indices=torch.tensor([1, 2, 4]),
            poison_delta=torch.arange(3 * 2, dtype=torch.float).view(3, 2),
            attacker=dict(attacker_id='left'),
        )
        right = dict(
            poison_indices=torch.tensor([2, 3, 5]),
            poison_delta=torch.arange(3 * 2, 6 * 2, dtype=torch.float).view(3, 2),
            attacker=dict(attacker_id='right'),
        )
        first_indices, first_delta, first_stats = resolve_dual_overlap(left, right, overlap_seed=11)
        second_indices, second_delta, second_stats = resolve_dual_overlap(left, right, overlap_seed=11)
        self.assertEqual(first_indices, second_indices)
        self.assertTrue(torch.equal(first_delta, second_delta))
        self.assertEqual(first_stats, second_stats)

    def test_resolve_dual_overlap_sum_clipped_keeps_shared_indices(self):
        left = dict(
            poison_indices=torch.tensor([1, 2]),
            poison_delta=torch.tensor([[0.8], [0.2]], dtype=torch.float),
            attacker=dict(attacker_id='left'),
        )
        right = dict(
            poison_indices=torch.tensor([2, 3]),
            poison_delta=torch.tensor([[0.7], [0.4]], dtype=torch.float),
            attacker=dict(attacker_id='right'),
        )

        merged_indices, merged_delta, stats = resolve_dual_overlap(
            left,
            right,
            overlap_seed=11,
            merge_rule='sum_clipped',
            eps=255,
            data_std=[1.0],
        )

        self.assertEqual(merged_indices, [1, 2, 3])
        self.assertTrue(torch.allclose(merged_delta, torch.tensor([[0.8], [0.9], [0.4]], dtype=torch.float)))
        self.assertEqual(stats['overlap_total'], 1)
        self.assertEqual(stats['lost_by_attacker'], {'left': 0, 'right': 0})
        self.assertEqual(stats['merge_rule'], 'sum_clipped')
        self.assertEqual(stats['effective_unique_poison_count'], 3)

    def test_randomized_deterministic_poison_ids_are_reproducible(self):
        class_ids = list(range(20))
        first = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=True)
        second = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=True)
        baseline = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=False)
        self.assertEqual(first, second)
        self.assertNotEqual(first, baseline)

    def test_poison_ids_seed_override_decouples_sampling_from_poisonkey(self):
        class_ids = list(range(20))
        first = select_deterministic_poison_ids(
            class_ids,
            5,
            '5-0-2731',
            randomize_poison_ids=True,
            poison_ids_seed='src5_rep0',
        )
        second = select_deterministic_poison_ids(
            class_ids,
            5,
            '5-3-2731',
            randomize_poison_ids=True,
            poison_ids_seed='src5_rep0',
        )
        self.assertEqual(first, second)

    def test_artifact_poison_indices_accepts_lists(self):
        poison_indices = _artifact_poison_indices([4, 1, 9])
        self.assertTrue(torch.equal(poison_indices, torch.tensor([4, 1, 9], dtype=torch.long)))

    def test_prepare_model_for_seed_uses_victim_seed_when_modelkey_is_fixed(self):
        class DummyModel:
            def __init__(self, args):
                self.args = args
                self.model_init_seed = None

            def initialize(self, seed=None):
                self.model_init_seed = self.args.modelkey if self.args.modelkey is not None else seed

        args = SimpleNamespace(scenario='from-scratch', modelkey=0)
        model = DummyModel(args)

        _prepare_model_for_seed(model, args, seed=7)

        self.assertEqual(model.model_init_seed, 7)
        self.assertEqual(args.modelkey, 0)

    def test_target_rows_include_c2_geometry_fields(self):
        class DummyNetwork:
            def eval(self):
                return self

            def __call__(self, target_images):
                return torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float)

        class DummyTrainset:
            classes = ['airplane', 'automobile', 'bird']

        class DummyKettle:
            trainset = DummyTrainset()
            targetset = [(torch.zeros(3, 2, 2), 0, 123)]
            setup = dict(device=torch.device('cpu'), dtype=torch.float)

        experiment = dict(experiment_id='c2_demo', family='C2')
        job = dict(
            job_id='dual_demo',
            pairing_id='pair_demo',
            condition='dog_vs_cat_own_aligned_easy',
            distance_bucket='near',
            source_pair_label='dog_vs_cat',
            source_stratum='near',
            source_source_rank=1,
            motif_label='own_aligned_easy',
            motif_rank=2,
            alignment_type='own_aligned',
            S=0.391,
            T=0.574,
            G=0.145,
            a_self=0.516,
            b_self=0.49,
            a_cross=0.569,
            b_cross=0.582,
            source_source_distance=0.391,
            target_target_distance=0.574,
            cross_alignment_gap=0.145,
            overlap_policy='assign_one_owner',
            merge_rule='assign_one_owner',
            overlap_seed=17,
        )
        attack_artifacts = [
            dict(
                artifact_path='/tmp/brew.pt',
                job_id='brew_src5_sel0_to7',
                target_adv_class=2,
                target_true_class=0,
                target_index=123,
                source_class=5,
                source_class_name='dog',
                target_true_class_name='airplane',
                target_adv_class_name='bird',
                attacker=dict(
                    attacker_id='src5_sel0_to7',
                    source_target_distance=0.516,
                    source_target_rank=1,
                ),
                brew_loss=0.5,
            )
        ]

        rows = _target_rows(
            experiment=experiment,
            job=job,
            run_type='dual',
            victim_seed=0,
            victim_run_id=0,
            network=DummyNetwork(),
            kettle=DummyKettle(),
            attack_artifacts=attack_artifacts,
            clean_accuracy=0.8,
            overlap_stats=dict(overlap_total=1, lost_by_attacker={'src5_sel0_to7': 1}),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['motif_label'], 'own_aligned_easy')
        self.assertEqual(row['source_stratum'], 'near')
        self.assertEqual(row['source_source_rank'], 1)
        self.assertEqual(row['source_pair_label'], 'dog_vs_cat')
        self.assertEqual(row['S'], 0.391)
        self.assertEqual(row['T'], 0.574)
        self.assertEqual(row['G'], 0.145)
        self.assertEqual(row['a_self'], 0.516)
        self.assertEqual(row['b_self'], 0.49)
        self.assertEqual(row['a_cross'], 0.569)
        self.assertEqual(row['b_cross'], 0.582)
        self.assertEqual(row['target_target_distance'], 0.574)
        self.assertEqual(row['cross_alignment_gap'], 0.145)
        self.assertEqual(row['alignment_type'], 'own_aligned')
        self.assertEqual(row['merge_rule'], 'assign_one_owner')

    def test_validate_brew_artifact_rejects_mismatched_poisonkey(self):
        experiment = dict(
            family='C1',
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                threatmodel='single-class',
                recipe='gradient-matching',
                budget=0.01,
                eps=8,
                attackoptim='signAdam',
                attackiter=250,
                init='randn',
                tau=0.1,
                restarts=8,
                loss='similarity',
                centreg=0,
                normreg=0,
                repel=0,
                pbatch=512,
                pshuffle=False,
                paugment=True,
                full_data=False,
                ensemble=1,
                stagger=None,
                max_epoch=None,
                targets=1,
                patch_size=8,
                data_aug='default',
                randomize_deterministic_poison_ids=True,
            ),
        )
        attacker = dict(
            attacker_id='src1_sel0_to0',
            poisonkey='1-0-100',
            poison_ids_seed='src1_rep0',
            brew_job_id='brew_src1_sel0_to0',
            selection_key=0,
            repeat_slot=0,
            target_index=100,
            source_class=1,
            target_true_class=1,
            target_adv_class=0,
        )
        artifact = dict(
            attacker=attacker,
            target_index=100,
            source_class=1,
            target_true_class=1,
            target_adv_class=0,
            brew_config=dict(args={**experiment['common_args'], 'poisonkey': '2-0-100', 'poison_ids_seed': 'src1_rep0'}),
        )
        expected_args = build_args_namespace(
            experiment['common_args'],
            dict(poisonkey='1-0-100', poison_ids_seed='src1_rep0', name='brew_src1_sel0_to0', targets=1, vruns=0),
        )
        with self.assertRaises(ValueError):
            validate_brew_artifact(experiment, artifact, attacker, expected_args)

    def test_validate_brew_artifact_rejects_mismatched_poison_ids_seed(self):
        experiment = dict(
            family='C2',
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                threatmodel='single-class',
                recipe='gradient-matching',
                budget=0.01,
                eps=8,
                attackoptim='signAdam',
                attackiter=250,
                init='randn',
                tau=0.1,
                restarts=8,
                loss='similarity',
                centreg=0,
                normreg=0,
                repel=0,
                pbatch=512,
                pshuffle=False,
                paugment=True,
                full_data=False,
                ensemble=1,
                stagger=None,
                max_epoch=None,
                targets=1,
                patch_size=8,
                data_aug='default',
                randomize_deterministic_poison_ids=True,
            ),
        )
        attacker = dict(
            attacker_id='src5_sel0_to3',
            poisonkey='5-3-500',
            poison_ids_seed='src5_rep0',
            brew_job_id='brew_src5_sel0_to3',
            selection_key=0,
            repeat_slot=0,
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=3,
        )
        artifact = dict(
            attacker={**attacker, 'poison_ids_seed': 'src5_rep1'},
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=3,
            brew_config=dict(args={**experiment['common_args'], 'poisonkey': '5-3-500', 'poison_ids_seed': 'src5_rep1'}),
        )
        expected_args = build_args_namespace(
            experiment['common_args'],
            dict(poisonkey='5-3-500', poison_ids_seed='src5_rep0', name='brew_src5_sel0_to3', targets=1, vruns=0),
        )
        with self.assertRaises(ValueError):
            validate_brew_artifact(experiment, artifact, attacker, expected_args)

    def test_validate_brew_artifact_accepts_legacy_c2_selection_key(self):
        experiment = dict(
            family='C2',
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                threatmodel='single-class',
                recipe='gradient-matching',
                budget=0.01,
                eps=8,
                attackoptim='signAdam',
                attackiter=250,
                init='randn',
                tau=0.1,
                restarts=8,
                loss='similarity',
                centreg=0,
                normreg=0,
                repel=0,
                pbatch=512,
                pshuffle=False,
                paugment=True,
                full_data=False,
                ensemble=1,
                stagger=None,
                max_epoch=None,
                targets=1,
                patch_size=8,
                data_aug='default',
                randomize_deterministic_poison_ids=True,
            ),
        )
        attacker = dict(
            attacker_id='src5_sel0_to3',
            poisonkey='5-3-500',
            poison_ids_seed='src5_rep0',
            brew_job_id='brew_src5_sel0_to3',
            repeat_slot=0,
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=3,
        )
        artifact = dict(
            attacker={**attacker, 'selection_key': 0},
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=3,
            brew_config=dict(args={**experiment['common_args'], 'poisonkey': '5-3-500', 'poison_ids_seed': 'src5_rep0'}),
        )
        expected_args = build_args_namespace(
            experiment['common_args'],
            dict(poisonkey='5-3-500', poison_ids_seed='src5_rep0', name='brew_src5_sel0_to3', targets=1, vruns=0),
        )

        validate_brew_artifact(experiment, artifact, attacker, expected_args)

    def test_try_reuse_brew_artifact_materializes_local_metadata_from_cache(self):
        common_args = dict(
            dataset='CIFAR10',
            net=['ResNet18'],
            scenario='from-scratch',
            threatmodel='single-class',
            recipe='gradient-matching',
            budget=0.01,
            eps=8,
            attackoptim='signAdam',
            attackiter=250,
            init='randn',
            tau=0.1,
            restarts=8,
            loss='similarity',
            centreg=0,
            normreg=0,
            repel=0,
            pbatch=512,
            pshuffle=False,
            paugment=True,
            full_data=False,
            ensemble=1,
            stagger=None,
            max_epoch=None,
            targets=1,
            patch_size=8,
            data_aug='default',
            randomize_deterministic_poison_ids=True,
        )
        experiment = dict(
            experiment_id='C2_new',
            family='C2',
            common_args=common_args,
        )
        attacker = dict(
            attacker_id='src5_sel0_to3_new',
            poisonkey='5-3-500',
            poison_ids_seed='src5_rep0',
            brew_job_id='brew_src5_sel0_to3_new',
            repeat_slot=0,
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job = dict(
                job_id='brew_src5_sel0_to3_new',
                attacker=attacker,
                arg_overrides=dict(
                    poisonkey='5-3-500',
                    poison_ids_seed='src5_rep0',
                    name='brew_src5_sel0_to3_new',
                    targets=1,
                ),
                artifact_path=os.path.join(tmpdir, 'local.pt'),
            )
            args = build_args_namespace(common_args, job['arg_overrides'])
            cache_path = _brew_cache_path(args, attacker, cache_dir=tmpdir)
            cached_artifact = dict(
                schema_version=1,
                experiment_id='C2_old',
                family='C2',
                job_id='brew_src5_sel0_to3',
                attacker=dict(
                    attacker_id='src5_sel0_to3',
                    poisonkey='5-3-500',
                    poison_ids_seed='src5_rep0',
                    brew_job_id='brew_src5_sel0_to3',
                    repeat_slot=0,
                    target_index=500,
                    source_class=5,
                    target_true_class=5,
                    target_adv_class=3,
                    selection_key=0,
                ),
                poison_delta=torch.zeros((1, 1), dtype=torch.float),
                poison_indices=torch.tensor([1], dtype=torch.long),
                target_index=500,
                target_image=torch.zeros((1,), dtype=torch.float),
                target_true_class=5,
                target_true_class_name='dog',
                target_adv_class=3,
                target_adv_class_name='cat',
                source_class=5,
                source_class_name='dog',
                brew_config=dict(args={**common_args, 'poisonkey': '5-3-500', 'poison_ids_seed': 'src5_rep0', 'name': 'brew_src5_sel0_to3', 'targets': 1}),
                brew_loss=0.0,
                poison_delta_norm=dict(mean_l2=0.0, max_l2=0.0, mean_linf=0.0, max_linf=0.0),
                timestamps=dict(train_time='0:00:01', brew_time='0:00:01'),
                clean_stats_available=False,
            )
            save_brew_artifact(cache_path, cached_artifact)

            reused_artifact = _try_reuse_brew_artifact(experiment, job, args, cache_dir=tmpdir)

            self.assertIsNotNone(reused_artifact)
            self.assertEqual(reused_artifact['attacker']['attacker_id'], attacker['attacker_id'])
            self.assertEqual(reused_artifact['job_id'], job['job_id'])
            self.assertTrue(os.path.isfile(job['artifact_path']))
            localized_artifact = load_brew_artifact(job['artifact_path'])
            self.assertEqual(localized_artifact['attacker']['attacker_id'], attacker['attacker_id'])
            self.assertEqual(localized_artifact['brew_fingerprint'], reused_artifact['brew_fingerprint'])

    def test_validate_brew_artifact_rejects_mismatched_explicit_poison_ids(self):
        experiment = dict(
            family='C5',
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                threatmodel='single-class',
                recipe='gradient-matching',
                budget=0.01,
                eps=8,
                attackoptim='signAdam',
                attackiter=250,
                init='randn',
                tau=0.1,
                restarts=8,
                loss='similarity',
                centreg=0,
                normreg=0,
                repel=0,
                pbatch=512,
                pshuffle=False,
                paugment=True,
                full_data=False,
                ensemble=1,
                stagger=None,
                max_epoch=None,
                targets=1,
                patch_size=8,
                data_aug='default',
                randomize_deterministic_poison_ids=True,
            ),
        )
        attacker = dict(
            attacker_id='src5_sel0_to0_ovl25',
            poisonkey='5-0-500',
            brew_job_id='brew_src5_sel0_to0_ovl25',
            selection_key=0,
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=0,
            explicit_poison_ids=[1, 2, 3],
        )
        artifact = dict(
            attacker={**attacker, 'explicit_poison_ids': [1, 2, 4]},
            target_index=500,
            source_class=5,
            target_true_class=5,
            target_adv_class=0,
            brew_config=dict(args={**experiment['common_args'], 'poisonkey': '5-0-500', 'explicit_poison_ids': [1, 2, 4]}),
        )
        expected_args = build_args_namespace(
            experiment['common_args'],
            dict(poisonkey='5-0-500', explicit_poison_ids=[1, 2, 3], name='brew_src5_sel0_to0_ovl25', targets=1, vruns=0),
        )
        with self.assertRaises(ValueError):
            validate_brew_artifact(experiment, artifact, attacker, expected_args)


class DualAttackSubmitterTests(unittest.TestCase):
    def setUp(self):
        self.experiment = dict(
            experiment_id='demo',
            scheduler=dict(account='aip-yiweilu', gpu='l40s', mem='20G', cpus=4, time='4:00:00'),
            output_root='/tmp/demo',
            brew_jobs=[dict(job_id='brew_a'), dict(job_id='brew_b')],
            solo_jobs=[
                dict(job_id='solo_a', attacker=dict(attacker_id='a', brew_job_id='brew_a')),
                dict(job_id='solo_b', attacker=dict(attacker_id='b', brew_job_id='brew_b')),
            ],
            dual_jobs=[
                dict(
                    job_id='dual_ab',
                    attackers=[
                        dict(attacker_id='a', brew_job_id='brew_a'),
                        dict(attacker_id='b', brew_job_id='brew_b'),
                    ],
                )
            ],
        )

    def test_plan_submission_specs_uses_solo_dependencies_for_dual_stage(self):
        specs = plan_submission_specs(
            self.experiment,
            '/tmp/demo.json',
            stage='all',
            dual_dependency_stage='solo',
            repo_root='/repo',
            python_executable='python3',
        )
        dual_spec = [spec for spec in specs if spec['job_id'] == 'dual_ab'][0]
        self.assertEqual(dual_spec['dependency_job_ids'], ['solo_a', 'solo_b'])

    def test_build_sbatch_command_uses_scheduler_and_dependencies(self):
        class Overrides:
            account = None
            gpu = None
            mem = None
            cpus = None
            time = None

        command = build_sbatch_command(
            dict(job_id='dual_ab', command=['python3', '/repo/run.py'], dependency_job_ids=['solo_a', 'solo_b']),
            self.experiment,
            Overrides(),
            dependency_slurm_ids=['101', '202'],
            output_dir='/tmp/slurm',
        )
        self.assertIn('--account', command)
        self.assertIn('aip-yiweilu', command)
        self.assertIn('--dependency', command)
        self.assertIn('afterok:101:202', command)


if __name__ == '__main__':
    unittest.main()
