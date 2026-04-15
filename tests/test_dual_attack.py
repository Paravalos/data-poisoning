import os
import tempfile
import unittest

import torch

from forest.data.kettle_det_experiment import select_deterministic_poison_ids
from forest.dual_attack.experiment import build_args_namespace, load_experiment, save_experiment
from forest.dual_attack.planners import build_c1_experiment
from forest.dual_attack.runtime import _artifact_poison_indices, resolve_dual_overlap, validate_brew_artifact
from forest.dual_attack.summary import summarize_experiment
from forest.dual_attack.submitter import build_sbatch_command, plan_submission_specs


class DualAttackPlannerTests(unittest.TestCase):
    def setUp(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.distance_matrix = torch.zeros((10, 10), dtype=torch.float)
        for row in range(10):
            for col in range(row + 1, 10):
                if row == 0 or col == 0:
                    value = col / 10.0
                else:
                    value = (row * 10 + col) / 100.0
                self.distance_matrix[row, col] = value
                self.distance_matrix[col, row] = value
        self.rankings = {
            'airplane': [
                dict(class_index=1, class_name='automobile', cosine_distance=0.1),
                dict(class_index=2, class_name='bird', cosine_distance=0.2),
                dict(class_index=3, class_name='cat', cosine_distance=0.3),
                dict(class_index=4, class_name='deer', cosine_distance=0.4),
                dict(class_index=5, class_name='dog', cosine_distance=0.5),
                dict(class_index=6, class_name='frog', cosine_distance=0.6),
                dict(class_index=7, class_name='horse', cosine_distance=0.7),
                dict(class_index=8, class_name='ship', cosine_distance=0.8),
                dict(class_index=9, class_name='truck', cosine_distance=0.9),
            ]
        }
        self.class_to_valid_indices = {class_idx: [class_idx * 100 + offset for offset in range(8)] for class_idx in range(10)}

    def test_build_c1_experiment_generates_expected_jobs(self):
        experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            planning_seed=17,
            repeats=3,
            victim_seeds=[0, 1],
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                randomize_deterministic_poison_ids=True,
            ),
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=7,
            distance_artifact_path='artifact.pt',
        )
        self.assertEqual(len(experiment['dual_jobs']), 9 * 3)
        self.assertEqual(len(experiment['solo_jobs']), len(experiment['brew_jobs']))
        self.assertEqual(experiment['class_names'][0], 'airplane')
        self.assertEqual(experiment['metadata']['pair_selection_strategy'], 'fixed_attacker_a_source_sweep')
        self.assertEqual(experiment['metadata']['planning_seed'], 17)
        self.assertTrue(experiment['common_args']['randomize_deterministic_poison_ids'])
        self.assertEqual(experiment['dual_jobs'][0]['attackers'][0]['source_class'], 5)
        self.assertEqual(experiment['dual_jobs'][0]['attackers'][1]['source_class'], 1)
        first_left = experiment['dual_jobs'][0]['attackers'][0]
        first_right = experiment['dual_jobs'][0]['attackers'][1]
        self.assertEqual(first_left['poisonkey'], f'5-0-{first_left["target_index"]}')
        self.assertEqual(first_right['poisonkey'], f'1-0-{first_right["target_index"]}')
        dog_pair_job = experiment['dual_jobs'][4 * 3]
        self.assertEqual(dog_pair_job['attackers'][0]['source_class'], 5)
        self.assertEqual(dog_pair_job['attackers'][1]['source_class'], 5)
        self.assertNotEqual(dog_pair_job['attackers'][0]['target_index'], dog_pair_job['attackers'][1]['target_index'])
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
            planning_seed=17,
            repeats=1,
            victim_seeds=[0],
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                randomize_deterministic_poison_ids=True,
            ),
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
        self.assertEqual(len(loaded['dual_jobs']), 9)
        self.assertIn('5', loaded['metadata']['sampled_target_indices'])

    def test_planning_seed_freezes_target_indices(self):
        base_kwargs = dict(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            repeats=2,
            victim_seeds=[0],
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                randomize_deterministic_poison_ids=True,
            ),
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        first = build_c1_experiment(planning_seed=3, **base_kwargs)
        second = build_c1_experiment(planning_seed=3, **base_kwargs)
        third = build_c1_experiment(planning_seed=4, **base_kwargs)
        self.assertEqual(first['metadata']['sampled_target_indices'], second['metadata']['sampled_target_indices'])
        self.assertNotEqual(first['metadata']['sampled_target_indices'], third['metadata']['sampled_target_indices'])

    def test_summary_uses_top_level_class_names(self):
        experiment = build_c1_experiment(
            experiment_id='C1_airplane_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            shared_target_class='airplane',
            fixed_attacker_a_source_class='dog',
            planning_seed=17,
            repeats=1,
            victim_seeds=[0],
            common_args=dict(
                dataset='CIFAR10',
                net=['ResNet18'],
                scenario='from-scratch',
                randomize_deterministic_poison_ids=True,
            ),
            output_root='artifacts/dual_attack',
            scheduler={},
            overlap_seed_base=0,
            distance_artifact_path='artifact.pt',
        )
        summary = summarize_experiment(experiment)
        self.assertIn('Target class: airplane (0)', summary)
        self.assertIn('Fixed attacker A source: dog (5)', summary)


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

    def test_randomized_deterministic_poison_ids_are_reproducible(self):
        class_ids = list(range(20))
        first = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=True)
        second = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=True)
        baseline = select_deterministic_poison_ids(class_ids, 5, '5-0-2731', randomize_poison_ids=False)
        self.assertEqual(first, second)
        self.assertNotEqual(first, baseline)

    def test_artifact_poison_indices_accepts_lists(self):
        poison_indices = _artifact_poison_indices([4, 1, 9])
        self.assertTrue(torch.equal(poison_indices, torch.tensor([4, 1, 9], dtype=torch.long)))

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
            brew_job_id='brew_src1_sel0_to0',
            selection_key=0,
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
            brew_config=dict(args={**experiment['common_args'], 'poisonkey': '2-0-100'}),
        )
        expected_args = build_args_namespace(
            experiment['common_args'],
            dict(poisonkey='1-0-100', name='brew_src1_sel0_to0', targets=1, vruns=0),
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
