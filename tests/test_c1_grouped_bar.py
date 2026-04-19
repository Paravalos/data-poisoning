import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_plot_module():
    module_path = Path(__file__).resolve().parents[1] / 'plotting' / 'c1_grouped_bar.py'
    spec = importlib.util.spec_from_file_location('c1_grouped_bar', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class C1GroupedBarTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_plot_module()

    def test_compute_grouped_bar_summary_aggregates_confidences(self):
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
                'dog_r1': solo_rows('dog_r1', 2, 'dog', 101, 0.8, [0.60, 0.64]),
                'dog_r2': solo_rows('dog_r2', 2, 'dog', 102, 0.8, [0.50, 0.54]),
                'dog_r3': solo_rows('dog_r3', 2, 'dog', 103, 0.8, [0.70, 0.74]),
                'cat_r1': solo_rows('cat_r1', 1, 'cat', 201, 0.2, [0.30, 0.34]),
                'cat_r2': solo_rows('cat_r2', 1, 'cat', 202, 0.2, [0.40, 0.44]),
                'cat_r3': solo_rows('cat_r3', 1, 'cat', 203, 0.2, [0.50, 0.54]),
            }
            for attacker_id, rows in solo_specs.items():
                _write_csv(repo_root / 'solo' / f'{attacker_id}.csv', rows)

            _write_csv(
                repo_root / 'dual' / 'cat_rep1.csv',
                dual_rows(
                    ('dog_r1', 2, 'dog', 101, 0.8, [0.80, 0.84]),
                    ('cat_r1', 1, 'cat', 201, 0.2, [0.20, 0.24]),
                    0.3,
                ),
            )
            _write_csv(
                repo_root / 'dual' / 'cat_rep2.csv',
                dual_rows(
                    ('dog_r2', 2, 'dog', 102, 0.8, [0.60, 0.64]),
                    ('cat_r2', 1, 'cat', 202, 0.2, [0.30, 0.34]),
                    0.3,
                ),
            )
            _write_csv(
                repo_root / 'dual' / 'cat_rep3.csv',
                dual_rows(
                    ('dog_r3', 2, 'dog', 103, 0.8, [0.50, 0.54]),
                    ('cat_r3', 1, 'cat', 203, 0.2, [0.60, 0.64]),
                    0.3,
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
                ],
            )
            with open(experiment_path, 'w', encoding='utf-8') as handle:
                json.dump(experiment, handle)

            original_repo_root = self.module.REPO_ROOT
            self.module.REPO_ROOT = repo_root
            try:
                summary_frame, fixed_name, target_name = self.module.compute_grouped_bar_summary(
                    experiment_path=experiment_path,
                )
                self.assertEqual(fixed_name, 'dog')
                self.assertEqual(target_name, 'airplane')
                self.assertEqual(len(summary_frame), 1)
                self.assertEqual(summary_frame.loc[0, 'partner_class_name'], 'cat')
                self.assertAlmostEqual(summary_frame.loc[0, 'a_solo_percentage_points'], 62.0)
                self.assertAlmostEqual(summary_frame.loc[0, 'b_solo_percentage_points'], 42.0)
                self.assertAlmostEqual(summary_frame.loc[0, 'a_together_percentage_points'], 65.33333333333333)
                self.assertAlmostEqual(summary_frame.loc[0, 'b_together_percentage_points'], 38.666666666666664)

                output_path = repo_root / 'bars.png'
                self.module.create_grouped_bar_plot(
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
