"""Polished interactive C2 faceted quadrant HTML — styled layout + rich hover."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _c2_loader import (  # noqa: E402
    MOTIF_ORDER,
    REPO_ROOT,
    SOURCE_STRATUM_ORDER,
    compute_c2_interaction_frame,
)

DEFAULT_EXPERIMENT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'c2_experiment.json'
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'artifacts' / 'c2' / 'plots' / 'c2_faceted_quadrant_polished.html'

MOTIF_TITLES = {
    'own_aligned_easy': 'own-aligned · easy',
    'own_aligned_far_apart': 'own-aligned · targets far apart',
    'cross_aligned_swapped': 'cross-aligned · swapped targets',
    'both_hard_far': 'both hard · targets far apart',
}
MOTIF_SUBTITLES = {
    'own_aligned_easy': 'both attackers attack "visually similar" classes',
    'own_aligned_far_apart': 'both easy, but T_A and T_B live far apart',
    'cross_aligned_swapped': 'each attacker points at the <i>other</i>’s natural target',
    'both_hard_far': 'both attacks are hard; victims are dissimilar',
}

STRATUM_COLORS = {
    'near': '#1f4e79',
    'mid_near': '#3f8fce',
    'medium': '#9aa0a6',
    'mid_far': '#ef7a3a',
    'far': '#b71c1c',
}
STRATUM_LABELS = {
    'near': 'near',
    'mid_near': 'mid-near',
    'medium': 'medium',
    'mid_far': 'mid-far',
    'far': 'far',
}

QUADRANT_TINTS = {
    'collab': 'rgba(56,142,60,0.08)',     # top-right: both gain (green)
    'mutual': 'rgba(183,28,28,0.08)',     # bottom-left: both lose (red)
    'asym_up': 'rgba(55,71,79,0.04)',     # top-left / bottom-right
}
QUADRANT_LABELS = {
    'collab': ('collaboration',  'rgba(46,125,50,0.85)'),
    'mutual': ('mutual degradation', 'rgba(183,28,28,0.85)'),
    'a_down_b_up': ('A weaker · B stronger', 'rgba(84,110,122,0.75)'),
    'a_up_b_down': ('A stronger · B weaker', 'rgba(84,110,122,0.75)'),
}


def _aggregate_with_labels(frame):
    grouping = ['motif_label', 'source_stratum']
    numeric_cols = ['i_a', 'i_b', 'i_sum', 'i_asym', 'target_target_distance',
                    'source_source_distance', 'a_self', 'b_self', 'cross_alignment_gap']
    label_cols = ['attacker_a_class_name', 'attacker_b_class_name',
                  'target_a_class_name', 'target_b_class_name']
    numeric = frame.groupby(grouping, observed=False)[numeric_cols].mean().reset_index()
    labels = frame.groupby(grouping, observed=False)[label_cols].first().reset_index()
    return numeric.merge(labels, on=grouping)


def _scale_sizes(values, min_size=18.0, max_size=42.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.max(), values.min()):
        return np.full(values.shape, (min_size + max_size) / 2.0)
    normalized = (values - values.min()) / (values.max() - values.min())
    return min_size + normalized * (max_size - min_size)


def _hover_html(row):
    def _pp(v): return f'<b>{v:+.2f}</b> pp'
    winner = ('dog' if row['i_a'] > row['i_b']
              else row['attacker_b_class_name'] if row['i_b'] > row['i_a']
              else 'tie')
    return (
        f"<span style='font-size:13px;font-weight:600;color:#263238'>"
        f"{row['attacker_a_class_name']} × {row['attacker_b_class_name']}"
        f"</span><br>"
        f"<span style='color:#607d8b'>{MOTIF_TITLES[row['motif_label']]} · {STRATUM_LABELS[row['source_stratum']]} d(A,B)</span>"
        f"<br><br>"
        f"Attacker A: <b>{row['attacker_a_class_name']}</b> → target <b>{row['target_a_class_name']}</b>  "
        f"I_A = {_pp(row['i_a'])}<br>"
        f"Attacker B: <b>{row['attacker_b_class_name']}</b> → target <b>{row['target_b_class_name']}</b>  "
        f"I_B = {_pp(row['i_b'])}<br>"
        f"<br>"
        f"d(A, B) = {row['source_source_distance']:.3f}<br>"
        f"d(T_A, T_B) = {row['target_target_distance']:.3f}<br>"
        f"d(A, T_A) = {row['a_self']:.3f} · d(B, T_B) = {row['b_self']:.3f}<br>"
        f"<span style='color:#546e7a'>winner: <b>{winner}</b></span>"
    )


def _add_quadrant_shading(fig, row, col, limit):
    # Collaboration (top-right)
    fig.add_shape(type='rect', x0=0, y0=0, x1=limit, y1=limit,
                  fillcolor=QUADRANT_TINTS['collab'], line=dict(width=0),
                  layer='below', row=row, col=col)
    # Mutual degradation (bottom-left)
    fig.add_shape(type='rect', x0=-limit, y0=-limit, x1=0, y1=0,
                  fillcolor=QUADRANT_TINTS['mutual'], line=dict(width=0),
                  layer='below', row=row, col=col)


def _add_quadrant_labels(fig, row, col, limit):
    axis = (row - 1) * 2 + col
    suffix = '' if axis == 1 else str(axis)
    labels = [
        (0.96 * limit, 0.96 * limit, 'right', 'top', *QUADRANT_LABELS['collab']),
        (-0.96 * limit, -0.96 * limit, 'left', 'bottom', *QUADRANT_LABELS['mutual']),
        (-0.96 * limit, 0.96 * limit, 'left', 'top', *QUADRANT_LABELS['a_down_b_up']),
        (0.96 * limit, -0.96 * limit, 'right', 'bottom', *QUADRANT_LABELS['a_up_b_down']),
    ]
    for x, y, xanchor, yanchor, text, color in labels:
        fig.add_annotation(
            xref=f'x{suffix}', yref=f'y{suffix}',
            x=x, y=y, text=f"<i>{text}</i>", showarrow=False,
            xanchor=xanchor, yanchor=yanchor,
            font=dict(color=color, size=10, family='Inter, Segoe UI, system-ui'),
        )


def _build_figure(summary):
    all_values = np.concatenate([summary['i_a'].to_numpy(), summary['i_b'].to_numpy()])
    limit = float(np.max(np.abs(all_values))) if all_values.size else 1.0
    limit = max(limit * 1.3, 1.0)

    summary = summary.assign(_marker_size=_scale_sizes(summary['target_target_distance'].to_numpy()))

    subplot_titles = [
        f"<b>{MOTIF_TITLES[m]}</b><br><span style='font-size:11px;color:#78909c'>{MOTIF_SUBTITLES[m]}</span>"
        for m in MOTIF_ORDER
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.09, vertical_spacing=0.16,
    )

    motif_positions = {m: (i // 2 + 1, i % 2 + 1) for i, m in enumerate(MOTIF_ORDER)}
    added_to_legend = set()

    for motif in MOTIF_ORDER:
        row, col = motif_positions[motif]
        _add_quadrant_shading(fig, row, col, limit)
        panel = summary[summary['motif_label'] == motif]

        for stratum in SOURCE_STRATUM_ORDER:
            subset = panel[panel['source_stratum'] == stratum]
            if subset.empty:
                continue
            show_legend = stratum not in added_to_legend
            added_to_legend.add(stratum)
            fig.add_trace(
                go.Scatter(
                    x=subset['i_a'], y=subset['i_b'],
                    mode='markers+text',
                    text=[f"{name}" for name in subset['attacker_b_class_name']],
                    textposition='top center',
                    textfont=dict(size=10, color='#37474f',
                                  family='Inter, Segoe UI, system-ui'),
                    marker=dict(
                        size=subset['_marker_size'],
                        color=STRATUM_COLORS[stratum],
                        line=dict(width=1.2, color='white'),
                        opacity=0.95,
                    ),
                    hovertext=[_hover_html(r) for _, r in subset.iterrows()],
                    hoverinfo='text',
                    hoverlabel=dict(
                        bgcolor='white', bordercolor=STRATUM_COLORS[stratum],
                        font=dict(family='Inter, Segoe UI, system-ui', size=12, color='#263238'),
                    ),
                    name=STRATUM_LABELS[stratum],
                    legendgroup=stratum,
                    showlegend=show_legend,
                ),
                row=row, col=col,
            )

        fig.add_hline(y=0, line=dict(color='#263238', width=1.2), row=row, col=col)
        fig.add_vline(x=0, line=dict(color='#263238', width=1.2), row=row, col=col)
        fig.add_shape(
            type='line', x0=-limit, y0=-limit, x1=limit, y1=limit,
            line=dict(color='#b0bec5', width=1, dash='dot'),
            layer='below', row=row, col=col,
        )
        _add_quadrant_labels(fig, row, col, limit)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(range=[-limit, limit], row=r, col=c,
                             showgrid=True, gridcolor='rgba(176,190,197,0.25)',
                             zeroline=False, ticks='outside', ticklen=4,
                             tickfont=dict(size=11, color='#455a64'))
            fig.update_yaxes(range=[-limit, limit], row=r, col=c,
                             showgrid=True, gridcolor='rgba(176,190,197,0.25)',
                             zeroline=False, ticks='outside', ticklen=4,
                             tickfont=dict(size=11, color='#455a64'))

    fig.update_xaxes(title_text='I_A (dog, pp)', row=2, col=1,
                     title_font=dict(size=12, color='#37474f'))
    fig.update_xaxes(title_text='I_A (dog, pp)', row=2, col=2,
                     title_font=dict(size=12, color='#37474f'))
    fig.update_yaxes(title_text='I_B (partner, pp)', row=1, col=1,
                     title_font=dict(size=12, color='#37474f'))
    fig.update_yaxes(title_text='I_B (partner, pp)', row=2, col=1,
                     title_font=dict(size=12, color='#37474f'))

    for ann in fig.layout.annotations[:len(MOTIF_ORDER)]:
        ann.font = dict(size=14, color='#263238', family='Inter, Segoe UI, system-ui')
        ann.yshift = 6

    fig.update_layout(
        margin=dict(l=60, r=30, t=40, b=60),
        plot_bgcolor='#fafbfc',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter, Segoe UI, system-ui', color='#263238'),
        legend=dict(
            title=dict(text='<b>d(A, B) stratum</b>', font=dict(size=12)),
            orientation='h',
            yanchor='top', y=-0.08,
            xanchor='center', x=0.5,
            bgcolor='rgba(255,255,255,0.0)',
            itemsizing='constant',
            font=dict(size=11),
        ),
        hovermode='closest',
        width=None, height=900,
    )
    return fig, limit


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>C2 — dual-attacker interaction</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg: #f4f6f8;
    --card: #ffffff;
    --ink: #1f2933;
    --muted: #52606d;
    --accent: #1f4e79;
    --border: #e4e7eb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 24px 56px;
    background: radial-gradient(1200px 600px at 10% -10%, #eef2f7 0%, var(--bg) 60%);
    color: var(--ink);
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{
    max-width: 1240px;
    margin: 0 auto;
  }}
  .hero {{
    margin-bottom: 20px;
  }}
  .eyebrow {{
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: 8px;
  }}
  h1 {{
    font-size: 30px; line-height: 1.15; margin: 0 0 10px; font-weight: 600;
    letter-spacing: -0.01em;
  }}
  .sub {{
    font-size: 15px; color: var(--muted); max-width: 820px; line-height: 1.55;
    margin: 0;
  }}
  .layout {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    gap: 24px;
    margin-top: 26px;
  }}
  @media (max-width: 1080px) {{
    .layout {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: var(--card);
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 6px rgba(15,23,42,0.04), 0 12px 32px rgba(15,23,42,0.04);
    padding: 12px 8px 4px;
  }}
  .legend {{
    padding: 20px 22px 22px;
    font-size: 13px; color: var(--ink); line-height: 1.55;
  }}
  .legend h3 {{
    margin: 0 0 10px; font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent); font-weight: 600;
  }}
  .legend p {{ margin: 0 0 12px; color: var(--muted); }}
  .legend .quad {{
    display: grid; grid-template-columns: 14px 1fr; gap: 10px;
    align-items: start; margin: 8px 0; font-size: 12.5px;
  }}
  .legend .dot {{ width: 12px; height: 12px; border-radius: 50%; margin-top: 3px; }}
  .kbd {{
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; background: #eef2f7; padding: 1px 6px; border-radius: 4px;
    color: #243b53;
  }}
  footer {{
    margin-top: 28px; font-size: 12px; color: var(--muted); text-align: right;
  }}
</style>
</head>
<body>
<div class="page">
  <div class="hero">
    <div class="eyebrow">experiment C2 · dual-attacker interaction</div>
    <h1>When does pairing two poisoning attackers help — or hurt?</h1>
    <p class="sub">
      Attacker A is always <b>dog</b>. Attacker B is a partner class whose feature-space distance from dog varies across five strata (near → far).
      Each panel is a different <i>target-geometry</i> motif. Points are averaged over 3 repeats.
      Hover for class assignments and raw interaction scores.
    </p>
  </div>

  <div class="layout">
    <div class="card">{plot_div}</div>
    <aside class="card legend">
      <h3>how to read it</h3>
      <p>I<sub>A</sub> and I<sub>B</sub> measure how much each attacker's adversarial confidence on its target changed when paired vs. run solo (percentage points).</p>
      <div class="quad"><span class="dot" style="background:#388e3c"></span><span><b>top-right</b> — both attacks get stronger together (collaboration).</span></div>
      <div class="quad"><span class="dot" style="background:#b71c1c"></span><span><b>bottom-left</b> — both weaker (mutual degradation).</span></div>
      <div class="quad"><span class="dot" style="background:#546e7a"></span><span><b>off-diagonal</b> — one attacker dominates the other (asymmetric).</span></div>
      <p style="margin-top:14px">Marker <b>color</b> encodes the d(A, B) stratum. Marker <b>size</b> encodes d(T<sub>A</sub>, T<sub>B</sub>) — larger means target classes live farther apart in feature space.</p>
      <p>Each panel label below the title describes the motif — see <span class="kbd">planners.py</span> for the selection rules.</p>
    </aside>
  </div>
  <footer>generated from <span class="kbd">artifacts/c2/c2_experiment.json</span></footer>
</div>
</body>
</html>
"""


def create_polished_quadrant_html(frame, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _aggregate_with_labels(frame).dropna(subset=['i_a', 'i_b'])
    fig, _ = _build_figure(summary)

    plot_div = pio.to_html(
        fig, include_plotlyjs='cdn', full_html=False,
        config=dict(displaylogo=False, responsive=True,
                    modeBarButtonsToRemove=['lasso2d', 'select2d', 'autoScale2d']),
    )
    html = HTML_TEMPLATE.format(plot_div=plot_div)
    output_path.write_text(html, encoding='utf-8')
    return output_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=str(DEFAULT_EXPERIMENT_PATH))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main():
    args = _build_argument_parser().parse_args()
    frame = compute_c2_interaction_frame(args.experiment)
    output_path = create_polished_quadrant_html(frame, args.output)
    print(f'Saved polished plot to {output_path}')


if __name__ == '__main__':
    main()
