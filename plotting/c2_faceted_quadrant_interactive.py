"""Interactive HTML version of the C2 faceted quadrant plot (hover for class names)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c2_loader import (  # noqa: E402
    MOTIF_ORDER,
    REPO_ROOT,
    SOURCE_STRATUM_ORDER,
    compute_c2_interaction_frame,
)

DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'c2_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'plots' / 'c2_faceted_quadrant_interactive.html'
DEFAULT_TITLE = 'C2: I_A vs I_B by motif (averaged over 3 repeats)'

STRATUM_COLORS = {
    'near': '#2166ac',
    'mid_near': '#67a9cf',
    'medium': '#737373',
    'mid_far': '#ef8a62',
    'far': '#b2182b',
}


def _aggregate_with_labels(frame):
    """Mean over repeats per (motif, stratum), keeping class-name labels (first observed)."""
    grouping = ['motif_label', 'source_stratum']
    numeric_cols = ['i_a', 'i_b', 'i_sum', 'i_asym', 'target_target_distance',
                    'source_source_distance', 'a_self', 'b_self', 'cross_alignment_gap']
    label_cols = ['attacker_a_class_name', 'attacker_b_class_name',
                  'target_a_class_name', 'target_b_class_name']
    numeric_summary = frame.groupby(grouping, observed=False)[numeric_cols].mean().reset_index()
    label_summary = frame.groupby(grouping, observed=False)[label_cols].first().reset_index()
    return numeric_summary.merge(label_summary, on=grouping)


def _hover_text(row):
    return (
        f"<b>{row['motif_label']} · {row['source_stratum']}</b><br>"
        f"Attacker A: <b>{row['attacker_a_class_name']}</b> → T_A: <b>{row['target_a_class_name']}</b><br>"
        f"Attacker B: <b>{row['attacker_b_class_name']}</b> → T_B: <b>{row['target_b_class_name']}</b><br>"
        f"I_A = {row['i_a']:+.2f} pp, I_B = {row['i_b']:+.2f} pp<br>"
        f"d(A, B) = {row['source_source_distance']:.3f}<br>"
        f"d(T_A, T_B) = {row['target_target_distance']:.3f}<br>"
        f"d(A, T_A) = {row['a_self']:.3f}, d(B, T_B) = {row['b_self']:.3f}"
    )


def _scale_sizes(values, min_size=14.0, max_size=34.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.max(), values.min()):
        return np.full(values.shape, (min_size + max_size) / 2.0)
    normalized = (values - values.min()) / (values.max() - values.min())
    return min_size + normalized * (max_size - min_size)


def create_interactive_quadrant_html(frame, output_path, *, title=DEFAULT_TITLE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _aggregate_with_labels(frame).dropna(subset=['i_a', 'i_b'])

    all_values = np.concatenate([summary['i_a'].to_numpy(), summary['i_b'].to_numpy()])
    limit = float(np.max(np.abs(all_values))) if all_values.size else 1.0
    limit = max(limit * 1.25, 1.0)

    sizes_all = _scale_sizes(summary['target_target_distance'].to_numpy())
    summary = summary.assign(_marker_size=sizes_all)

    subplot_titles = list(MOTIF_ORDER)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    motif_positions = {motif: (i // 2 + 1, i % 2 + 1) for i, motif in enumerate(MOTIF_ORDER)}
    seen_stratum_for_legend = set()

    for motif in MOTIF_ORDER:
        panel = summary[summary['motif_label'] == motif]
        row, col = motif_positions[motif]
        for stratum in SOURCE_STRATUM_ORDER:
            subset = panel[panel['source_stratum'] == stratum]
            if subset.empty:
                continue
            show_legend = stratum not in seen_stratum_for_legend
            seen_stratum_for_legend.add(stratum)
            fig.add_trace(
                go.Scatter(
                    x=subset['i_a'], y=subset['i_b'],
                    mode='markers+text',
                    text=subset['attacker_b_class_name'],
                    textposition='top center',
                    textfont=dict(size=10),
                    marker=dict(
                        size=subset['_marker_size'],
                        color=STRATUM_COLORS[stratum],
                        line=dict(width=1, color='black'),
                        opacity=0.85,
                    ),
                    hovertext=[_hover_text(r) for _, r in subset.iterrows()],
                    hoverinfo='text',
                    name=stratum,
                    legendgroup=stratum,
                    showlegend=show_legend,
                ),
                row=row, col=col,
            )

        fig.add_hline(y=0, line=dict(color='black', width=1), row=row, col=col)
        fig.add_vline(x=0, line=dict(color='black', width=1), row=row, col=col)
        fig.add_shape(
            type='line', x0=-limit, y0=-limit, x1=limit, y1=limit,
            line=dict(color='black', width=0.8, dash='dash'),
            row=row, col=col,
        )
        annotations = [
            dict(x=0.95 * limit, y=0.95 * limit, text='collaboration', showarrow=False,
                 xanchor='right', yanchor='top', font=dict(color='gray', size=9)),
            dict(x=-0.95 * limit, y=-0.95 * limit, text='mutual degradation', showarrow=False,
                 xanchor='left', yanchor='bottom', font=dict(color='gray', size=9)),
            dict(x=-0.95 * limit, y=0.95 * limit, text='A down, B up', showarrow=False,
                 xanchor='left', yanchor='top', font=dict(color='gray', size=9)),
            dict(x=0.95 * limit, y=-0.95 * limit, text='A up, B down', showarrow=False,
                 xanchor='right', yanchor='bottom', font=dict(color='gray', size=9)),
        ]
        for ann in annotations:
            fig.add_annotation(xref=f'x{_axis_suffix(row, col)}',
                               yref=f'y{_axis_suffix(row, col)}', **ann)

    fig.update_xaxes(range=[-limit, limit], title_text='I_A (attacker A, pp)',
                     row=2, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(range=[-limit, limit], title_text='I_A (attacker A, pp)',
                     row=2, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(range=[-limit, limit], row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(range=[-limit, limit], row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(range=[-limit, limit], title_text='I_B (partner, pp)',
                     row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(range=[-limit, limit], title_text='I_B (partner, pp)',
                     row=2, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(range=[-limit, limit], row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(range=[-limit, limit], row=2, col=2, showgrid=True, gridcolor='lightgray')

    fig.update_layout(
        title=dict(text=title, x=0.5),
        legend=dict(title='d(A, B) stratum', itemsizing='constant'),
        width=1100, height=900,
        plot_bgcolor='white',
        hovermode='closest',
    )
    fig.write_html(str(output_path), include_plotlyjs='cdn', full_html=True)
    return output_path


def _axis_suffix(row, col):
    idx = (row - 1) * 2 + col
    return '' if idx == 1 else str(idx)


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main():
    args = _build_argument_parser().parse_args()
    frame = compute_c2_interaction_frame(args.experiment)
    output_path = create_interactive_quadrant_html(frame, args.output)
    print(f'Saved interactive plot to {output_path}')


if __name__ == '__main__':
    main()
