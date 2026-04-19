import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PLOTTING_DIR = Path(__file__).resolve().parents[1] / 'plotting'


def _load_module(name, filename):
    module_path = PLOTTING_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _solo_rows(attacker_id, confidences):
    return [dict(attacker_id=attacker_id, adv_confidence=value) for value in confidences]


def _dual_rows(a_id, a_conf, b_id, b_conf):
    rows = [dict(attacker_id=a_id, adv_confidence=v) for v in a_conf]
    rows.extend(dict(attacker_id=b_id, adv_confidence=v) for v in b_conf)
    return rows


STRATA = ['near', 'mid_near', 'medium', 'mid_far', 'far']
MOTIFS = ['own_aligned_easy', 'own_aligned_far_apart', 'cross_aligned_swapped', 'both_hard_far']


def _write_c2_fixture(repo_root):
    solo_dir = repo_root / 'solo'
    dual_dir = repo_root / 'dual'
    solo_dir.mkdir()
    dual_dir.mkdir()

    solo_jobs = []
    dual_jobs = []
    # Two attackers share all dual combinations — keeps the fixture tiny.
    a_id = 'src5_sel0_toX'
    b_id = 'src3_sel0_toY'
    _write_csv(solo_dir / f'{a_id}.csv', _solo_rows(a_id, [0.30, 0.30]))
    _write_csv(solo_dir / f'{b_id}.csv', _solo_rows(b_id, [0.40, 0.40]))
    solo_jobs.append(dict(attacker=dict(attacker_id=a_id), output_path=f'solo/{a_id}.csv'))
    solo_jobs.append(dict(attacker=dict(attacker_id=b_id), output_path=f'solo/{b_id}.csv'))

    for stratum_idx, stratum in enumerate(STRATA):
        for motif_idx, motif in enumerate(MOTIFS):
            # Vary dual adv_confidence so geometry coefficients are estimable.
            a_dual = 0.30 + 0.02 * stratum_idx + 0.01 * motif_idx
            b_dual = 0.40 + 0.015 * stratum_idx - 0.005 * motif_idx
            pairing_id = f'{stratum}_{motif}'
            csv_path = dual_dir / f'{pairing_id}.csv'
            _write_csv(csv_path, _dual_rows(a_id, [a_dual, a_dual], b_id, [b_dual, b_dual]))
            dual_jobs.append(dict(
                pairing_id=pairing_id,
                source_pair_label='dog_vs_cat',
                source_stratum=stratum,
                motif_label=motif,
                alignment_type='own_aligned',
                source_source_distance=0.3 + 0.05 * stratum_idx,
                target_target_distance=0.5 + 0.02 * motif_idx,
                a_self=0.4 + 0.01 * motif_idx,
                b_self=0.45 + 0.01 * motif_idx,
                a_cross=0.5 + 0.01 * stratum_idx,
                b_cross=0.55 + 0.01 * stratum_idx,
                cross_alignment_gap=0.1 * (motif_idx - 1.5),
                output_path=f'dual/{pairing_id}.csv',
                attackers=[
                    dict(attacker_id=a_id, source_class=5),
                    dict(attacker_id=b_id, source_class=3),
                ],
            ))

    experiment = dict(
        class_names=['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        metadata=dict(),
        solo_jobs=solo_jobs,
        dual_jobs=dual_jobs,
    )
    experiment_path = repo_root / 'experiment.json'
    with open(experiment_path, 'w', encoding='utf-8') as handle:
        json.dump(experiment, handle)
    return experiment_path


class C2LoaderTests(unittest.TestCase):
    def test_compute_interaction_frame_fields_and_math(self):
        loader = _load_module('_c2_loader', '_c2_loader.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = _write_c2_fixture(repo_root)
            frame = loader.compute_c2_interaction_frame(experiment_path, repo_root=repo_root)

            self.assertEqual(len(frame), len(STRATA) * len(MOTIFS))
            self.assertEqual(set(frame['source_stratum'].astype(str).unique()), set(STRATA))
            self.assertEqual(set(frame['motif_label'].astype(str).unique()), set(MOTIFS))

            near_easy = frame[(frame['source_stratum'] == 'near') & (frame['motif_label'] == 'own_aligned_easy')].iloc[0]
            # a_dual - a_solo = 0.30 - 0.30 = 0.00 -> 0 pp
            # b_dual - b_solo = 0.40 - 0.40 = 0.00 -> 0 pp
            self.assertAlmostEqual(near_easy['i_a'], 0.0, places=6)
            self.assertAlmostEqual(near_easy['i_b'], 0.0, places=6)

            far_hard = frame[(frame['source_stratum'] == 'far') & (frame['motif_label'] == 'both_hard_far')].iloc[0]
            # stratum_idx=4, motif_idx=3
            # a_dual = 0.30 + 0.02*4 + 0.01*3 = 0.41 -> I_A = (0.41 - 0.30)*100 = 11 pp
            # b_dual = 0.40 + 0.015*4 - 0.005*3 = 0.445 -> I_B = 4.5 pp
            self.assertAlmostEqual(far_hard['i_a'], 11.0, places=5)
            self.assertAlmostEqual(far_hard['i_b'], 4.5, places=5)
            self.assertAlmostEqual(far_hard['i_sum'], 15.5, places=5)
            self.assertAlmostEqual(far_hard['i_asym'], 6.5, places=5)

    def test_aggregate_by_cell_returns_expected_shape(self):
        loader = _load_module('_c2_loader', '_c2_loader.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = _write_c2_fixture(repo_root)
            frame = loader.compute_c2_interaction_frame(experiment_path, repo_root=repo_root)
            summary = loader.aggregate_by_cell(frame, ['i_sum', 'i_asym'])
            self.assertEqual(len(summary), len(STRATA) * len(MOTIFS))
            self.assertIn('i_sum', summary.columns)
            self.assertIn('i_asym', summary.columns)


class C2FacetedQuadrantTests(unittest.TestCase):
    def test_plot_is_saved(self):
        loader = _load_module('_c2_loader', '_c2_loader.py')
        plot_module = _load_module('c2_faceted_quadrant', 'c2_faceted_quadrant.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = _write_c2_fixture(repo_root)
            frame = loader.compute_c2_interaction_frame(experiment_path, repo_root=repo_root)
            output_path = repo_root / 'quadrant.png'
            plot_module.create_faceted_quadrant_plot(frame, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class C2HeatmapGridTests(unittest.TestCase):
    def test_plot_is_saved(self):
        loader = _load_module('_c2_loader', '_c2_loader.py')
        plot_module = _load_module('c2_heatmap_grid', 'c2_heatmap_grid.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = _write_c2_fixture(repo_root)
            frame = loader.compute_c2_interaction_frame(experiment_path, repo_root=repo_root)
            output_path = repo_root / 'heatmap.png'
            plot_module.create_heatmap_grid_plot(frame, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class C2RegressionCoefficientsTests(unittest.TestCase):
    def test_fit_and_plot(self):
        loader = _load_module('_c2_loader', '_c2_loader.py')
        plot_module = _load_module('c2_regression_coefficients', 'c2_regression_coefficients.py')
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            experiment_path = _write_c2_fixture(repo_root)
            frame = loader.compute_c2_interaction_frame(experiment_path, repo_root=repo_root)

            coeffs, model = plot_module.fit_standardized_ols(frame, 'i_sum')
            self.assertEqual(len(coeffs), len(plot_module.FEATURE_COLUMNS))
            self.assertIn('coefficient', coeffs.columns)
            self.assertIn('ci_low', coeffs.columns)
            self.assertIn('ci_high', coeffs.columns)
            self.assertTrue((coeffs['ci_low'] <= coeffs['coefficient']).all())
            self.assertTrue((coeffs['coefficient'] <= coeffs['ci_high']).all())
            self.assertEqual(int(model.nobs), len(frame))

            output_path = repo_root / 'regression.png'
            plot_module.create_regression_coefficient_plot(frame, output_path)
            self.assertTrue(output_path.exists())


if __name__ == '__main__':
    unittest.main()
