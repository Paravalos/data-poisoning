import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[1] / 'plotting' / 'c1_interaction_scatter.py'
    spec = importlib.util.spec_from_file_location('c1_interaction_scatter', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class C1InteractionScatterTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_plot_module()

    def test_compute_interaction_summary_aggregates_confidence_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = repo_root / 'experiment.json'
            (repo_root / 'solo').mkdir()
            (repo_root / 'dual').mkdir()

            def solo_rows(attacker_id, source_class, source_name, target_index, distance, confidences):
                return [
                    dict(
                        attacker_id=attacker_id,
                        source_class=source_class,
                        source_class_name=source_name,
                        target_index=target_index,
                        source_target_distance=distance,
                        adv_confidence=value,
                    )
                    for value in confidences
                ]

            def dual_rows(left, right, source_source_distance):
                rows = []
                for attacker_id, source_class, source_name, target_index, target_distance, confidences in (left, right):
                    rows.extend([
                        dict(
                            attacker_id=attacker_id,
                            source_class=source_class,
                            source_class_name=source_name,
                            target_index=target_index,
                            source_target_distance=target_distance,
                            source_source_distance=source_source_distance,
                            adv_confidence=value,
                        )
                        for value in confidences
                    ])
                return rows

            solo_specs = {
                'dog_r1': solo_rows('dog_r1', 2, 'dog', 101, 0.8, [0.6, 0.6]),
                'dog_r2': solo_rows('dog_r2', 2, 'dog', 102, 0.8, [0.5, 0.5]),
                'dog_r3': solo_rows('dog_r3', 2, 'dog', 103, 0.8, [0.5, 0.5]),
                'cat_r1': solo_rows('cat_r1', 1, 'cat', 201, 0.2, [0.4, 0.4]),
                'cat_r2': solo_rows('cat_r2', 1, 'cat', 202, 0.2, [0.35, 0.35]),
                'cat_r3': solo_rows('cat_r3', 1, 'cat', 203, 0.2, [0.5, 0.5]),
                'dog_self': solo_rows('dog_self', 2, 'dog', 301, 0.8, [0.4, 0.4]),
            }
            for attacker_id, rows in solo_specs.items():
                _write_csv(repo_root / 'solo' / f'{attacker_id}.csv', rows)

            _write_csv(
                repo_root / 'dual' / 'cat_rep1.csv',
                dual_rows(
                    ('dog_r1', 2, 'dog', 101, 0.8, [0.7, 0.7]),
                    ('cat_r1', 1, 'cat', 201, 0.2, [0.5, 0.5]),
                    0.3,
                ),
            )
            _write_csv(
                repo_root / 'dual' / 'cat_rep2.csv',
                dual_rows(
                    ('dog_r2', 2, 'dog', 102, 0.8, [0.4, 0.4]),
                    ('cat_r2', 1, 'cat', 202, 0.2, [0.25, 0.25]),
                    0.3,
                ),
            )
            _write_csv(
                repo_root / 'dual' / 'cat_rep3.csv',
                dual_rows(
                    ('dog_r3', 2, 'dog', 103, 0.8, [0.8, 0.8]),
                    ('cat_r3', 1, 'cat', 203, 0.2, [0.7, 0.7]),
                    0.3,
                ),
            )
            _write_csv(
                repo_root / 'dual' / 'dog_self.csv',
                dual_rows(
                    ('dog_r1', 2, 'dog', 101, 0.8, [0.65, 0.65]),
                    ('dog_self', 2, 'dog', 301, 0.8, [0.45, 0.45]),
                    0.0,
                ),
            )

            experiment = dict(
                class_names=['airplane', 'cat', 'dog'],
                metadata=dict(
                    fixed_attacker_a_source_class=2,
                    shared_target_class=0,
                ),
                solo_jobs=[
                    dict(attacker=dict(attacker_id='dog_r1'), output_path='solo/dog_r1.csv'),
                    dict(attacker=dict(attacker_id='dog_r2'), output_path='solo/dog_r2.csv'),
                    dict(attacker=dict(attacker_id='dog_r3'), output_path='solo/dog_r3.csv'),
                    dict(attacker=dict(attacker_id='cat_r1'), output_path='solo/cat_r1.csv'),
                    dict(attacker=dict(attacker_id='cat_r2'), output_path='solo/cat_r2.csv'),
                    dict(attacker=dict(attacker_id='cat_r3'), output_path='solo/cat_r3.csv'),
                    dict(attacker=dict(attacker_id='dog_self'), output_path='solo/dog_self.csv'),
                ],
                dual_jobs=[
                    dict(
                        pairing_id='fixed_dog_vs_cat_rep1',
                        source_source_distance=0.3,
                        output_path='dual/cat_rep1.csv',
                        attackers=[
                            dict(attacker_id='dog_r1', source_class=2),
                            dict(attacker_id='cat_r1', source_class=1),
                        ],
                    ),
                    dict(
                        pairing_id='fixed_dog_vs_cat_rep2',
                        source_source_distance=0.3,
                        output_path='dual/cat_rep2.csv',
                        attackers=[
                            dict(attacker_id='dog_r2', source_class=2),
                            dict(attacker_id='cat_r2', source_class=1),
                        ],
                    ),
                    dict(
                        pairing_id='fixed_dog_vs_cat_rep3',
                        source_source_distance=0.3,
                        output_path='dual/cat_rep3.csv',
                        attackers=[
                            dict(attacker_id='dog_r3', source_class=2),
                            dict(attacker_id='cat_r3', source_class=1),
                        ],
                    ),
                    dict(
                        pairing_id='fixed_dog_vs_dog_rep1',
                        source_source_distance=0.0,
                        output_path='dual/dog_self.csv',
                        attackers=[
                            dict(attacker_id='dog_r1', source_class=2),
                            dict(attacker_id='dog_self', source_class=2),
                        ],
                    ),
                ],
            )
            with open(experiment_path, 'w', encoding='utf-8') as handle:
                json.dump(experiment, handle)

            original_repo_root = self.module.REPO_ROOT
            self.module.REPO_ROOT = repo_root
            try:
                summary_frame, per_image_frame, fixed_name, target_name = self.module.compute_interaction_summary(
                    experiment_path=experiment_path,
                )
                self.assertEqual(fixed_name, 'dog')
                self.assertEqual(target_name, 'airplane')
                self.assertEqual(len(summary_frame), 1)
                self.assertEqual(summary_frame.loc[0, 'partner_class_name'], 'cat')
                self.assertAlmostEqual(summary_frame.loc[0, 'i_a_percentage_points'], 10.0)
                self.assertAlmostEqual(summary_frame.loc[0, 'i_b_percentage_points'], 20.0 / 3.0)
                self.assertAlmostEqual(summary_frame.loc[0, 'dog_partner_distance'], 0.3)
                self.assertAlmostEqual(summary_frame.loc[0, 'partner_target_distance'], 0.2)
                self.assertEqual(summary_frame.loc[0, 'repeats'], 3)
                self.assertEqual(len(per_image_frame), 3)

                output_path = repo_root / 'scatter.png'
                self.module.create_interaction_scatter_plot(
                    summary_frame,
                    output_path,
                    fixed_attacker_name=fixed_name,
                    target_class_name=target_name,
                )
                self.assertTrue(output_path.exists())
            finally:
                self.module.REPO_ROOT = original_repo_root


if __name__ == '__main__':
    unittest.main()
