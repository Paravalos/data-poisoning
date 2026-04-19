"""C2 standardized OLS regression of interaction on geometry features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c2_loader import REPO_ROOT, compute_c2_interaction_frame  # noqa: E402

DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'c2_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'plots' / 'c2_regression_coefficients.png'
DEFAULT_TITLE = 'C2: standardized effect of geometry on interaction (OLS, 95% CI)'

FEATURE_COLUMNS = (
    'source_source_distance',
    'target_target_distance',
    'a_self',
    'b_self',
    'cross_alignment_gap',
)
FEATURE_DISPLAY = {
    'source_source_distance': 'd(A, B)',
    'target_target_distance': 'd(T_A, T_B)',
    'a_self': 'd(A, T_A)',
    'b_self': 'd(B, T_B)',
    'cross_alignment_gap': 'cross-alignment gap',
}
OUTCOMES = (
    ('i_sum', 'I_A + I_B (cooperation)'),
    ('i_a', 'I_A (attacker A)'),
    ('i_b', 'I_B (attacker B)'),
)


def _standardize(series):
    values = series.to_numpy(dtype=float)
    std = values.std(ddof=0)
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def fit_standardized_ols(frame, outcome_col, feature_columns=FEATURE_COLUMNS):
    design = pd.DataFrame({col: _standardize(frame[col]) for col in feature_columns})
    design = sm.add_constant(design, has_constant='add')
    y = frame[outcome_col].to_numpy(dtype=float)
    model = sm.OLS(y, design).fit()
    ci = model.conf_int(alpha=0.05)
    rows = []
    for feature in feature_columns:
        rows.append(dict(
            feature=feature,
            coefficient=float(model.params[feature]),
            ci_low=float(ci.loc[feature, 0]),
            ci_high=float(ci.loc[feature, 1]),
            p_value=float(model.pvalues[feature]),
        ))
    return pd.DataFrame(rows), model


def create_regression_coefficient_plot(frame, output_path, *, title=DEFAULT_TITLE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fits = {label: fit_standardized_ols(frame, outcome)[0]
            for outcome, label in OUTCOMES}

    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(5.5 * len(OUTCOMES), 5), sharey=True)
    if len(OUTCOMES) == 1:
        axes = [axes]

    y_positions = np.arange(len(FEATURE_COLUMNS))[::-1]
    feature_labels = [FEATURE_DISPLAY[f] for f in FEATURE_COLUMNS]

    for ax, (outcome, label) in zip(axes, OUTCOMES):
        coeffs = fits[label]
        coeffs = coeffs.set_index('feature').reindex(list(FEATURE_COLUMNS)).reset_index()
        centers = coeffs['coefficient'].to_numpy()
        lows = coeffs['ci_low'].to_numpy()
        highs = coeffs['ci_high'].to_numpy()
        errors = np.vstack([centers - lows, highs - centers])

        colors = ['#b2182b' if c > 0 else '#2166ac' for c in centers]
        ax.errorbar(centers, y_positions, xerr=errors, fmt='none',
                    ecolor='black', elinewidth=1.1, capsize=4)
        ax.scatter(centers, y_positions, c=colors, s=110, edgecolor='black', linewidth=0.7, zorder=3)
        ax.axvline(0.0, color='black', linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_labels)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('standardized coefficient (pp per 1 SD of feature)')
        ax.grid(True, axis='x', linestyle=':', linewidth=0.6, alpha=0.5)

        for y, p in zip(y_positions, coeffs['p_value'].to_numpy()):
            marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if marker:
                ax.text(ax.get_xlim()[1], y, ' ' + marker, ha='left', va='center',
                        fontsize=10, color='black')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main():
    args = _build_argument_parser().parse_args()
    frame = compute_c2_interaction_frame(args.experiment)
    for outcome, label in OUTCOMES:
        coeffs, model = fit_standardized_ols(frame, outcome)
        print(f'--- {label} (R^2 = {model.rsquared:.3f}, n = {int(model.nobs)}) ---')
        print(coeffs.to_string(index=False))
    output_path = create_regression_coefficient_plot(frame, args.output)
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
