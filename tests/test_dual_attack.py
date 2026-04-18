import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from forest.data.kettle_det_experiment import select_deterministic_poison_ids
from forest.dual_attack.eval import _prepare_model_for_seed, _target_rows
from forest.dual_attack.experiment import build_args_namespace, load_experiment, save_experiment
from forest.dual_attack.planners import build_c1_experiment, build_c2_experiment
from forest.dual_attack.runtime import _artifact_poison_indices, resolve_dual_overlap, validate_brew_artifact
from forest.dual_attack.summary import summarize_experiment
from forest.dual_attack.submitter import build_sbatch_command, plan_submission_specs


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
            planning_seed=17,
            repeats=3,
            victim_seeds=[0, 1],
            common_args=self.common_args,
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
            common_args=self.common_args,
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
            planning_seed=17,
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
            planning_seed=17,
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
        target_indices = {attacker['target_index'] for attacker in dog_attackers}
        poison_ids_seeds = {attacker['poison_ids_seed'] for attacker in dog_attackers}
        self.assertEqual(len(target_indices), 1)
        self.assertEqual(poison_ids_seeds, {'src5_rep0'})

    def test_build_c2_experiment_reuses_same_seed_for_same_attacker_across_pairings(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            planning_seed=17,
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

    def test_c2_summary_uses_top_level_class_names(self):
        experiment = build_c2_experiment(
            experiment_id='C2_dog_target_sweep_v1',
            class_names=self.class_names,
            rankings=self.rankings,
            distance_matrix=self.distance_matrix,
            class_to_valid_indices=self.class_to_valid_indices,
            fixed_attacker_a_source_class='dog',
            planning_seed=17,
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
