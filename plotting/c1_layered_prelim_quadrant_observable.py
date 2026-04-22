"""Interactive C1 layered HTML figure built with Observable Plot.

Produces a self-contained HTML file with a light theme, a trend panel for
attack-vector alignment, and the cooperation quadrant diagnostic. Loads Plot +
D3 from CDN, no installs required.

Usage:
    python plotting/c1_layered_prelim_quadrant_observable.py
    open -a "Google Chrome" artifacts/c1/plots/c1_layered_prelim_quadrant_observable.html
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from c1_layered_prelim_quadrant_interactive import build_cell_frame, OUT as _ORIG_OUT  # noqa
from _c2_loader import compute_c2_interaction_frame  # noqa

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'artifacts' / 'c1' / 'plots' / 'c1_layered_prelim_quadrant_observable.html'
C2_EXPERIMENT_PATH = REPO / 'artifacts' / 'c2' / 'c2_experiment.json'
DOG_CAT_TRUCK_ROOT = REPO / 'artifacts' / 'dual_attack' / 'dog_cat_vs_truck_auto' / 'dog_cat_vs_truck_auto'
FORK_FAR_APART_ROOT = REPO / 'artifacts' / 'dual_attack' / 'fork_far_apart' / 'fork_far_apart'
C2_MINI_ROOT = REPO / 'artifacts' / 'c2' / 'c2_mini' / 'c2_mini'
C6_EXPERIMENT_PATH = REPO / 'artifacts' / 'c6' / 'c6_target_pair_similarity.json'
C6_ROOT = REPO / 'artifacts' / 'c6' / 'c6_target_pair_similarity'
DIST_PATH = REPO / 'artifacts' / 'c1' / 'class_distance_matrix_cifar10_resnet18_modelkey0.pt'

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>C1 layered - alignment explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/dist/plot.umd.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root {
    --bg:        #f6f8fb;
    --panel:     #ffffff;
    --ink:       #17202a;
    --muted:     #5f6b7a;
    --rule:      #dce3ec;
    --grid:      #e7edf4;
    --coop:      #e5f4e9;
    --suppress:  #fde8eb;
    --asym-a:    #fff1dc;
    --asym-b:    #e5f0fb;
    --blue:      #2b6cb0;
    --red:       #c2414d;
    --green:     #2f8f46;
    --line:      #1f2937;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-feature-settings: 'cv11', 'ss01';
    -webkit-font-smoothing: antialiased;
  }
  header {
    padding: 28px 32px 20px;
    max-width: 1500px;
    margin: 0 auto;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: flex-start;
  }
  h1 {
    margin: 0 0 6px;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0;
  }
  header p {
    margin: 0;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
  }
  .button {
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #ffffff;
    color: var(--ink);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    min-height: 34px;
    padding: 6px 12px;
  }
  .button:hover { border-color: #9fb0c4; }
  .button.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #ffffff;
  }
  .button.secondary {
    background: #e9f0f7;
    border-color: #c8d5e3;
  }
  .view-controls {
    display: inline-flex;
    flex: 0 0 auto;
    gap: 6px;
    padding: 4px;
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #eef4fa;
  }
  .header-actions {
    align-items: flex-end;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .dataset-controls {
    display: inline-flex;
    gap: 6px;
    padding: 4px;
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #ffffff;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    gap: 20px;
    padding: 0 32px 40px;
    max-width: 1500px;
    margin: 0 auto;
  }
  body.table-mode .layout { grid-template-columns: 1fr; }
  body.table-mode #side { display: none; }
  body.full-mode .layout { grid-template-columns: 1fr; }
  body.full-mode #side { display: none; }
  .card {
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  }
  .card-head {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: baseline;
    padding: 18px 20px 0;
  }
  .card-head h2 {
    margin: 0;
    color: var(--ink);
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0;
  }
  .card-head p {
    margin: 6px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
  }
  .plot-controls {
    align-items: flex-end;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .segmented {
    display: inline-flex;
    gap: 6px;
    padding: 4px;
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #f8fafc;
  }
  .stat {
    flex: 0 0 auto;
    color: var(--ink);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    padding: 5px 8px;
    border: 1px solid var(--rule);
    border-radius: 6px;
    background: #f8fafc;
  }
  .is-hidden { display: none !important; }
  .plot-panel { padding: 8px 20px 18px; overflow: hidden; }
  #plot { padding-top: 20px; }
  #trend svg, #plot svg { display: block; margin: 0 auto; overflow: visible; }
  #table-wrap, #filter-table-wrap { padding: 8px 20px 22px; overflow-x: auto; }
  .table-unit {
    color: var(--muted);
    font-size: 12px;
    padding: 12px 20px 0;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    min-width: 980px;
    font-size: 13px;
  }
  th, td {
    border-bottom: 1px solid var(--rule);
    padding: 9px 10px;
    text-align: left;
    vertical-align: middle;
    white-space: nowrap;
  }
  th {
    color: var(--muted);
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    user-select: none;
  }
  th:hover { color: var(--ink); }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: #f8fafc; }
  td.num {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .table-tools {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
  }
  .column-picker {
    position: relative;
  }
  .column-picker summary {
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #ffffff;
    color: var(--ink);
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    list-style: none;
    min-height: 34px;
    padding: 7px 12px;
  }
  .column-picker summary::-webkit-details-marker { display: none; }
  .column-picker summary::after {
    color: var(--muted);
    content: ' v';
    font-size: 11px;
    font-weight: 500;
  }
  .column-picker[open] summary {
    border-color: #9fb0c4;
  }
  .subtle-picker summary {
    background: transparent;
    border-color: transparent;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    min-height: auto;
    padding: 2px 4px;
  }
  .subtle-picker summary:hover,
  .subtle-picker[open] summary {
    border-color: var(--rule);
    color: var(--ink);
  }
  .subtle-picker .column-panel {
    min-width: 196px;
  }
  .subtle-picker .column-option {
    justify-content: flex-start;
  }
  .column-panel {
    background: #ffffff;
    border: 1px solid var(--rule);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
    display: grid;
    gap: 4px;
    min-width: 220px;
    padding: 8px;
    position: absolute;
    right: 0;
    top: calc(100% + 6px);
    z-index: 10;
  }
  .column-option {
    align-items: center;
    border-radius: 6px;
    color: var(--ink);
    display: flex;
    gap: 8px;
    font-size: 13px;
    padding: 6px 8px;
    white-space: nowrap;
  }
  .column-option:hover { background: #f3f6fa; }
  .column-option input { margin: 0; }
  .filter-grid {
    display: grid;
    gap: 16px;
    padding: 16px 20px 0;
  }
  .filter-group {
    display: grid;
    gap: 8px;
  }
  .filter-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
  }
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .filter-chip {
    border: 1px solid transparent;
    border-radius: 8px;
    background: #eef3f8;
    color: var(--ink);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    min-height: 32px;
    padding: 5px 10px;
  }
  .filter-chip.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #ffffff;
  }
  .motif-controls {
    max-width: 560px;
  }
  .motif-controls .filter-label {
    margin-bottom: 6px;
    text-align: right;
  }
  .motif-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
    margin-top: 8px;
    max-width: 560px;
    text-align: right;
  }
  .filter-summary {
    color: var(--muted);
    font-size: 13px;
    padding: 12px 20px 0;
  }
  .table-tools label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
  }
  select {
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #ffffff;
    color: var(--ink);
    font: inherit;
    font-size: 13px;
    min-height: 34px;
    padding: 5px 30px 5px 10px;
  }
  #side { padding: 24px 24px; min-height: 520px; }
  #side h2 {
    margin: 0 0 14px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    color: var(--muted);
  }
  .placeholder {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
  }
  .attacker-stack {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 2px 0 14px;
  }
  .attacker-route {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid var(--rule);
    border-radius: 8px;
    background: #ffffff;
  }
  .attacker-route-main {
    align-items: center;
    display: inline-flex;
    gap: 8px;
    min-width: 0;
  }
  .attacker-role {
    align-items: center;
    background: #e9f0f7;
    border: 1px solid #d3deeb;
    border-radius: 6px;
    display: inline-flex;
    height: 28px;
    justify-content: center;
    width: 28px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
    color: var(--ink);
  }
  .attacker-name {
    font-size: 19px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .attacker-arrow {
    color: var(--muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 13px;
  }
  .attacker-target {
    font-size: 18px;
    font-weight: 600;
    color: #354152;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .role-sub {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .detail-summary {
    color: var(--ink);
    font-size: 13px;
    line-height: 1.45;
    margin: 8px 0 14px;
  }
  .detail-summary strong {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 13px;
  }
  .detail-card {
    background: transparent;
    border-top: 1px solid var(--rule);
    border-radius: 0;
    padding: 13px 0 0;
    margin: 13px 0 0;
  }
  .detail-card .block-label {
    margin: 0 0 10px;
  }
  .formula {
    font-size: 13px;
    color: var(--muted);
    margin: -4px 0 10px;
    line-height: 1.45;
  }
  .formula .katex { font-size: 1em; }
  .kv .katex { font-size: 1em; }
  .value-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 12.5px;
    background: #eef3f9;
    color: var(--ink);
  }
  .value-pill.pos { background: #e2f3e8; color: #12642b; }
  .value-pill.neg { background: #fce8eb; color: #a61b2a; }
  .value-pill .pm {
    color: var(--muted);
    font-size: 11px;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    color: white;
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .qtag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0;
  }
  .q-coop     { background: var(--coop);     color: #12642b; }
  .q-suppress { background: var(--suppress); color: #a61b2a; }
  .q-asym-a   { background: var(--asym-a);   color: #96500e; }
  .q-asym-b   { background: var(--asym-b);   color: #1f5b99; }

  .n-chip {
    display: inline-block;
    margin-left: 8px;
    color: var(--muted);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
  }
  hr.sep { border: 0; border-top: 1px solid var(--rule); margin: 16px 0; }
  .block-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .kv {
    display: grid;
    grid-template-columns: 140px 1fr;
    row-gap: 6px;
    column-gap: 12px;
    font-size: 13px;
    line-height: 1.5;
  }
  .kv .k {
    color: var(--muted);
  }
  .kv .v {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 12.5px;
  }
  .legend {
    display: flex;
    gap: 18px;
    padding: 0 20px 16px;
    font-size: 13px;
    color: var(--muted);
    align-items: center;
    flex-wrap: wrap;
  }
  .legend .dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }
  .val-pos { color: #12642b; }
  .val-neg { color: #a61b2a; }
  .arrow   { color: var(--muted); margin: 0 4px; }
  @media (max-width: 1100px) {
    .layout { grid-template-columns: 1fr; }
    .topbar, .card-head { flex-direction: column; }
    .header-actions { align-items: flex-start; }
    .motif-controls .filter-label { text-align: left; }
    .motif-description { text-align: left; }
    .view-controls { flex-wrap: wrap; }
    .plot-controls, .table-tools { align-items: flex-start; }
  }
</style>
</head>
<body>

<header>
  <div class="topbar">
    <div>
      <h1>C1 layered - alignment explorer</h1>
      <p>Dual adversarial attacks summarized as pair-level interaction cells. Each point is one attacker pair or target-pair condition averaged over repetitions. Plot colors and the trend x-axis use the selected alignment or distance diagnostic.</p>
    </div>
    <div class="header-actions">
      <div class="dataset-controls" aria-label="Dataset selector">
        <button class="button active" id="dataset-c1-button" type="button">C1</button>
        <button class="button" id="dataset-c2-button" type="button">C2</button>
        <button class="button" id="dataset-c2-mini-button" type="button">C2 mini</button>
        <button class="button" id="dataset-c6-button" type="button">C6</button>
      </div>
      <div class="motif-controls is-hidden" id="motif-controls">
        <div class="filter-label" id="motif-filter-label">Motif filter</div>
        <div class="chip-row" id="motif-filter"></div>
        <div class="motif-description" id="motif-description">Click motifs to filter C2.</div>
      </div>
      <div class="view-controls" aria-label="Main view selector">
        <button class="button active" id="plot-view-button" type="button">Plot view</button>
        <button class="button" id="table-view-button" type="button">Table view</button>
        <button class="button" id="filter-view-button" type="button">Filtered view</button>
      </div>
      <div class="view-controls" aria-label="Success metric selector">
        <button class="button active" id="metric-confidence-button" type="button">Target confidence</button>
        <button class="button" id="metric-success-button" type="button">Success rate</button>
      </div>
    </div>
  </div>
</header>

<div class="layout">
  <main>
    <section class="card" id="plot-card">
    <div class="card-head">
      <div>
        <h2 id="plot-title">At least one attacker usually degrades</h2>
        <p id="plot-subtitle">x = cos(attack<sub>A</sub>, attack<sub>B</sub>), y = worst single change min(I<sub>A</sub>, I<sub>B</sub>). Values below zero mean at least one attacker does worse than solo.</p>
      </div>
      <div class="plot-controls">
        <div class="segmented" aria-label="Plot selector">
          <button class="button active" id="quadrant-button" type="button">Quadrant</button>
          <button class="button" id="trend-button" type="button">Trend</button>
        </div>
        <details class="column-picker subtle-picker" id="alignment-picker-wrap">
          <summary>Alignment</summary>
          <div class="column-panel" id="alignment-controls"></div>
        </details>
        <div class="stat" id="trend-stat"></div>
      </div>
    </div>
    <div class="plot-panel" id="trend"></div>
    <div class="plot-panel is-hidden" id="plot"></div>
    <div class="legend is-hidden" id="quadrant-legend">
      <span id="colorbar"></span>
      <span style="color:var(--muted)">uniform point size</span>
    </div>
    </section>

    <section class="card is-hidden" id="table-card">
      <div class="card-head">
        <div>
          <h2>Cell table</h2>
          <p>Sorted summaries for every attacker pair. Click any column header or use the sort menu.</p>
        </div>
        <div class="table-tools">
          <label for="sort-select">Sort by</label>
          <select id="sort-select">
            <option value="anchorDrop">Worst A drop</option>
            <option value="partnerDrop">Worst B drop</option>
            <option value="meanDrop">Worst mean change</option>
            <option value="alignmentDesc">Highest alignment</option>
            <option value="alignmentAsc">Lowest alignment</option>
	        </select>
          <details class="column-picker">
            <summary>Columns</summary>
            <div class="column-panel" id="table-column-controls"></div>
          </details>
        </div>
      </div>
      <div class="table-unit" id="table-unit"></div>
      <div id="table-wrap"></div>
    </section>

    <section class="card is-hidden" id="filter-card">
      <div class="card-head">
        <div>
          <h2>Filtered table</h2>
          <p>Select one or more values for Attacker A, Attacker B, and target class. Empty selections include all values.</p>
        </div>
        <div class="table-tools">
          <details class="column-picker">
            <summary>Columns</summary>
            <div class="column-panel" id="filter-column-controls"></div>
          </details>
          <button class="button" id="clear-filters" type="button">Clear filters</button>
        </div>
      </div>
      <div class="filter-grid">
        <div class="filter-group">
          <div class="filter-label">Attacker A</div>
          <div class="chip-row" id="filter-anchor"></div>
        </div>
        <div class="filter-group">
          <div class="filter-label">Attacker B</div>
          <div class="chip-row" id="filter-partner"></div>
        </div>
        <div class="filter-group">
          <div class="filter-label">Target class</div>
          <div class="chip-row" id="filter-target"></div>
        </div>
      </div>
      <div class="filter-summary" id="filter-summary"></div>
      <div class="table-unit" id="filter-table-unit"></div>
      <div id="filter-table-wrap"></div>
    </section>
  </main>

  <div class="card" id="side">
    <h2>Details</h2>
    <div id="placeholder" class="placeholder">Click any point to inspect that cell.<br><br>The four quadrants mean:
      <ul style="margin:10px 0 0 18px;padding:0;color:var(--muted);line-height:1.9">
        <li><span style="color:#12642b;font-weight:600">upper-right</span> - mutual cooperation</li>
        <li><span style="color:#a61b2a;font-weight:600">lower-left</span> - mutual degradation</li>
        <li><span style="color:#96500e;font-weight:600">lower-right</span> - A improves, B degrades</li>
        <li><span style="color:#1f5b99;font-weight:600">upper-left</span> - A degrades, B improves</li>
      </ul>
    </div>
    <div id="detail" style="display:none"></div>
  </div>
</div>

<script>
const DATASETS = __DATASETS_JSON__;

const ROLE = { bird: 'near', cat: 'mid', frog: 'far' };
const ANCHOR_COLORS = {
  bird: '#2b6cb0', cat: '#c2414d', frog: '#2f8f46',
  dog: '#2b6cb0', deer: '#2f8f46', horse: '#7c3aed',
  ship: '#0f766e', truck: '#b45309', airplane: '#64748b',
};
const FALLBACK_COLORS = ['#2b6cb0', '#c2414d', '#2f8f46', '#7c3aed', '#0f766e', '#b45309', '#64748b'];
const P = Plot;

let currentDatasetName = 'C1';
let currentMetric = 'confidence';
let allRows = [];
let rows = [];
let colorScale = null;
let trend = null;
let currentPlotMode = 'quadrant';
let useDarkPointStroke = false;
const currentAlignmentModeByDataset = {
  C1: 'attack',
  C2: 'attack',
  C2mini: 'attack',
  C6: 'target_pair',
};
const METRIC_LABEL = {
  success: 'Δ success vs solo',
  confidence: 'Δ target confidence vs solo',
};
const METRIC_UNIT = {
  success: 'percentage points',
  confidence: 'raw confidence score',
};
const activeMotifs = new Set();
const MOTIF_LABELS = {
  own_aligned_easy: 'Aligned easy targets',
  own_aligned_far_apart: 'Aligned targets far apart',
  cross_aligned_swapped: 'Swapped targets',
  both_hard_far: 'Hard targets far apart',
  c2_mini_easy_close: 'Mini easy close',
  c2_mini_easy_far: 'Mini easy far',
  c2_mini_hard_close: 'Mini hard close',
  c2_mini_hard_far: 'Mini hard far',
  fork_far_apart: 'Same-source fork',
  c6_closest: 'Closest target images',
  c6_median: 'Median target images',
  c6_q75: '75th-percentile target images',
};
const MOTIF_DESCRIPTIONS = {
  c2_mini_easy_close: 'Dog -> cat and frog -> bird. Both source-to-target jumps are easy, and the two targets are close.',
  c2_mini_easy_far: 'Dog -> cat and frog -> truck. One easy dog target is paired with a farther frog target.',
  c2_mini_hard_close: 'Dog -> truck and frog -> automobile. Both jumps are harder, with close vehicle targets.',
  c2_mini_hard_far: 'Dog -> ship and frog -> horse. Both jumps are harder, and the two targets are farther apart.',
  fork_far_apart: 'Frog runs two same-source attacks at once, forked toward horse and ship, which are far apart in feature space.',
  c6_closest: 'Two target images from the same true class are chosen from the closest clean-representation pair for that class.',
  c6_median: 'Two target images from the same true class are chosen near the median clean-representation distance for that class.',
  c6_q75: 'Two target images from the same true class are chosen near the 75th-percentile clean-representation distance for that class.',
};
const ALIGNMENT_MODES = {
  C1: {
    attack: {
      key: 'cos_attack',
      axisLabel: 'cos(δ_A, δ_B)',
      titleLabel: 'attack-direction alignment',
      detailLabel: 'cos(δ_A, δ_B)',
      subtitleText: 'attack-direction alignment',
    },
    source: {
      key: 'cos_source',
      axisLabel: 'cos(A, B)',
      titleLabel: 'source-class cosine',
      detailLabel: 'cos(A, B)',
      subtitleText: 'source-class cosine',
    },
  },
  C2: {
    cross: {
      key: 'cross_alignment_gap',
      axisLabel: 'cross-alignment gap',
      titleLabel: 'cross-target alignment gap',
      detailLabel: 'cross-alignment gap',
      subtitleText: 'cross-target alignment gap',
    },
    attack: {
      key: 'cos_attack_dir',
      axisLabel: 'cos(δ_A, δ_B)',
      titleLabel: 'attack-direction alignment',
      detailLabel: 'cos(δ_A, δ_B)',
      subtitleText: 'attack-direction alignment',
    },
    source: {
      key: 'cos_source',
      axisLabel: 'cos(A, B)',
      titleLabel: 'source-class cosine',
      detailLabel: 'cos(A, B)',
      subtitleText: 'source-class cosine',
    },
  },
};
ALIGNMENT_MODES.C2mini = ALIGNMENT_MODES.C2;
ALIGNMENT_MODES.C6 = {
  target_pair: {
    key: 'target_target_distance',
    axisLabel: 'd(target image A, target image B)',
    titleLabel: 'target-pair distance',
    detailLabel: 'd(target image A, target image B)',
    subtitleText: 'target-image pair distance',
  },
  gradient: {
    key: 'gradient_cosine',
    axisLabel: 'cos(gradient A, gradient B)',
    titleLabel: 'target-gradient cosine',
    detailLabel: 'cos(gradient A, gradient B)',
    subtitleText: 'target-gradient cosine',
  },
  source_target: {
    key: 'd_anc_tgt',
    axisLabel: 'd(source class, poison class)',
    titleLabel: 'source-to-poison class distance',
    detailLabel: 'd(source class, poison class)',
    subtitleText: 'source-to-poison class distance',
  },
};

function isC2LikeDataset() {
  return currentDatasetName === 'C2' || currentDatasetName === 'C2mini';
}

function isMotifDataset() {
  return isC2LikeDataset() || currentDatasetName === 'C6';
}

function prepareRows(data) {
  const suf = currentMetric === 'confidence' ? 'c' : 's';
  const scale = currentMetric === 'confidence' ? 1 : 100;
  return data.map((d, index) => {
    const ia = d[`I_a_${suf}`];
    const ib = d[`I_b_${suf}`];
    const iaSem = d[`I_a_${suf}_sem`];
    const ibSem = d[`I_b_${suf}_sem`];
    return {
      ...d,
      row_id: d.row_id || `${currentDatasetName}-${index}`,
      target_class: d.target_class || 'airplane',
      target_classes: d.target_classes || [d.target_class || 'airplane'],
      'Attacker A': d.anchor,
      'Attacker B': d.partner,
      'Target class': d.target_class || 'airplane',
      Ia:     ia * scale,
      Ib:     ib * scale,
      Isem_a: (iaSem || 0) * scale,
      Isem_b: (ibSem || 0) * scale,
      Isum:   (ia + ib) * scale,
      Imean:  (ia + ib) * scale / 2,
      Iworst: Math.min(ia, ib) * scale,
    };
  });
}

function metricDigits() {
  return currentMetric === 'confidence' ? 3 : 1;
}

function metricUnit() {
  return METRIC_UNIT[currentMetric];
}

function metricAxisUnit() {
  return currentMetric === 'confidence' ? 'raw score change' : 'percentage points';
}

function metricShortLabel() {
  return currentMetric === 'confidence' ? 'Δ target confidence' : 'Δ success';
}

function formatMetricValue(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(metricDigits())}`;
}

function metricLimit(data = rows) {
  if (currentMetric !== 'confidence') {
    return 20;
  }
  const values = data.flatMap(d => [d.Ia, d.Ib]);
  const maxAbs = Math.max(0.05, ...values.map(v => Math.abs(v)));
  return Math.ceil((maxAbs * 1.25) / 0.05) * 0.05;
}

function quadrantRects(limit) {
  return [
    { x: 0,      y: 0,      w:  limit, h:  limit, fill: '#e5f4e9' },
    { x: -limit, y: -limit, w:  limit, h:  limit, fill: '#fde8eb' },
    { x: 0,      y: -limit, w:  limit, h:  limit, fill: '#fff1dc' },
    { x: -limit, y: 0,      w:  limit, h:  limit, fill: '#e5f0fb' },
  ];
}

function attackerColor(name) {
  if (ANCHOR_COLORS[name]) return ANCHOR_COLORS[name];
  const names = [...new Set(rows.map(row => row.anchor))].sort();
  const idx = Math.max(0, names.indexOf(name));
  return FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
}

function regression(data, xAccessor, yAccessor) {
  const n = data.length;
  const xs = data.map(xAccessor);
  const ys = data.map(yAccessor);
  const xbar = d3.mean(xs);
  const ybar = d3.mean(ys);
  if (n < 2) {
    return { slope: 0, intercept: ybar || 0, r: null };
  }
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xbar;
    const dy = ys[i] - ybar;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  if (sxx === 0 || syy === 0) {
    return { slope: 0, intercept: ybar || 0, r: null };
  }
  const slope = sxy / sxx;
  const intercept = ybar - slope * xbar;
  const r = sxy / Math.sqrt(sxx * syy);
  return { slope, intercept, r };
}

function defaultPointStroke() {
  return useDarkPointStroke ? '#17202a' : '#ffffff';
}

function currentAlignmentConfig() {
  const options = ALIGNMENT_MODES[currentDatasetName];
  const selected = currentAlignmentModeByDataset[currentDatasetName];
  if (options && options[selected]) return options[selected];
  return {
    key: 'cross_alignment_gap',
    axisLabel: DATASETS[currentDatasetName].alignment_label,
    titleLabel: 'alignment',
    detailLabel: DATASETS[currentDatasetName].alignment_label,
    subtitleText: 'alignment',
  };
}

function alignmentValue(d) {
  const value = d[currentAlignmentConfig().key];
  return value == null || !Number.isFinite(value) ? 0 : value;
}

function renderAlignmentControls() {
  const wrap = document.getElementById('alignment-picker-wrap');
  const options = ALIGNMENT_MODES[currentDatasetName];
  wrap.classList.toggle('is-hidden', !options);
  if (!options) {
    wrap.removeAttribute('open');
    document.getElementById('alignment-controls').innerHTML = '';
    return;
  }
  document.getElementById('alignment-controls').innerHTML = Object.entries(options).map(([mode, config]) => {
    const checked = mode === currentAlignmentModeByDataset[currentDatasetName] ? 'checked' : '';
    return `<label class="column-option"><input type="radio" name="alignment-mode" value="${mode}" ${checked}>${config.detailLabel}</label>`;
  }).join('');
  document.querySelectorAll('input[name="alignment-mode"]').forEach(input => {
    input.addEventListener('change', () => {
      if (!input.checked || input.value === currentAlignmentModeByDataset[currentDatasetName]) return;
      currentAlignmentModeByDataset[currentDatasetName] = input.value;
      renderPlots();
    });
  });
}

function plotCopy() {
  const dataset = DATASETS[currentDatasetName].label;
  const metricLabel = METRIC_LABEL[currentMetric];
  const metricName = currentMetric === 'confidence' ? 'target confidence' : 'success rate';
  const alignment = currentAlignmentConfig();
  return {
    trend: {
      title: `${dataset}: alignment vs cooperation (${metricName})`,
      subtitle: `${metricShortLabel()}, ${metricUnit()}. x = ${alignment.subtitleText}, y = I_A + I_B. Values below zero mean the pair does worse than solo.`,
      stat: trend.r == null
        ? 'trend unavailable for fewer than two distinct points'
        : `r=${trend.r.toFixed(2)}  slope=${trend.slope.toFixed(metricDigits())}/${alignment.detailLabel}`,
    },
    quadrant: {
      title: `${dataset}: cooperation quadrant (${metricName})`,
      subtitle: `${metricShortLabel()}, ${metricUnit()}. x = I_A, y = I_B. Color = ${alignment.subtitleText}.`,
      stat: `${rows.length} cells`,
    },
  };
}

function buildTrendChart() {
  if (!rows.length) return emptyPlot('No rows available for this dataset yet.');
  const alignment = currentAlignmentConfig();
  const alignmentValues = rows.map(alignmentValue);
  const alignmentMin = Math.min(...alignmentValues);
  const alignmentMax = Math.max(...alignmentValues);
  const trendPad = Math.max(0.02, (alignmentMax - alignmentMin) * 0.08);
  const trendX = [alignmentMin - trendPad, alignmentMax + trendPad];
  const trendLine = trendX.map(x => ({ x, y: trend.slope * x + trend.intercept }));
  const yValues = rows.map(r => r.Isum);
  const minPad = currentMetric === 'confidence' ? 0.02 : 3;
  const yPad = Math.max(minPad, (Math.max(...yValues) - Math.min(...yValues)) * 0.12);

  return P.plot({
    width: 900,
    height: 430,
    marginLeft: 74,
    marginBottom: 58,
    marginTop: 24,
    marginRight: 34,
    x: {
      domain: trendX,
      label: alignment.axisLabel,
      labelOffset: 42,
      grid: true,
      nice: false,
      tickSize: 4,
    },
    y: {
      domain: [Math.min(...yValues) - yPad, Math.max(...yValues) + yPad],
      label: 'I_A + I_B',
      labelOffset: 48,
      grid: true,
      nice: false,
      tickSize: 4,
    },
    style: {
      background: 'transparent',
      fontFamily: 'Inter, system-ui, sans-serif',
      fontSize: '12px',
      color: '#17202a',
    },
    marks: [
      P.ruleY([0], { stroke: '#7c8795', strokeWidth: 1, strokeDasharray: '3,4' }),
      P.line(trendLine, {
        x: 'x',
        y: 'y',
        stroke: '#1f2937',
        strokeWidth: 2.4,
        strokeOpacity: 0.9,
      }),
      P.dot(rows, {
        x: alignmentValue,
        y: 'Isum',
        r: 12,
        fill: d => colorScale(alignmentValue(d)),
        stroke: defaultPointStroke,
        strokeWidth: useDarkPointStroke ? 2.5 : 2,
        fillOpacity: 0.95,
        channels: {
          'Attacker A': 'Attacker A',
          'Attacker B': 'Attacker B',
        },
        tip: {
          format: {
            x: false,
            y: false,
            fill: false,
            'Attacker A': true,
            'Attacker B': true,
          },
        },
      }),
    ],
  });
}

function buildQuadrantChart() {
  if (!rows.length) return emptyPlot('No rows available for this dataset yet.');
  const limit = metricLimit();
  return P.plot({
    width: 900,
    height: 640,
    marginLeft: 70,
    marginBottom: 60,
    marginTop: 36,
    marginRight: 36,
    x: { domain: [-limit, limit], label: 'I_A', labelOffset: 44,
         grid: true, nice: false, tickSize: 4 },
    y: { domain: [-limit, limit], label: 'I_B', labelOffset: 44,
         grid: true, nice: false, tickSize: 4 },
    style: {
      background: 'transparent',
      fontFamily: 'Inter, system-ui, sans-serif',
      fontSize: '12px',
      color: '#17202a',
    },
    marks: [
      P.rect(quadrantRects(limit), { x1: 'x', y1: 'y', x2: d => d.x + d.w, y2: d => d.y + d.h,
                      fill: 'fill', fillOpacity: 0.46 }),
      P.ruleX([0], { stroke: '#5b6675', strokeWidth: 1, strokeOpacity: 0.75 }),
      P.ruleY([0], { stroke: '#5b6675', strokeWidth: 1, strokeOpacity: 0.75 }),
      P.link([{x1: -limit, y1: -limit, x2: limit, y2: limit}], {
        x1: 'x1', y1: 'y1', x2: 'x2', y2: 'y2',
        stroke: '#8b96a6', strokeDasharray: '3,5', strokeWidth: 1,
      }),
      P.text(
        [
          {x:  limit*0.95, y:  limit*0.95, t: 'mutual cooperation', c: '#12642b'},
          {x: -limit*0.95, y: -limit*0.95, t: 'mutual degradation', c: '#a61b2a'},
          {x:  limit*0.95, y: -limit*0.95, t: 'A improves, B degrades', c: '#96500e'},
          {x: -limit*0.95, y:  limit*0.95, t: 'A degrades, B improves', c: '#1f5b99'},
        ],
        { x: 'x', y: 'y', text: 't', fill: 'c', fontWeight: 600, fontSize: 11,
          textAnchor: d => d.x > 0 ? 'end' : 'start' }
      ),
      P.dot(rows, {
        x: 'Ia', y: 'Ib',
        r: 11.33,
        fill: d => colorScale(alignmentValue(d)),
        stroke: defaultPointStroke,
        strokeWidth: useDarkPointStroke ? 2.5 : 2,
        fillOpacity: 0.95,
        channels: {
          'Attacker A': 'Attacker A',
          'Attacker B': 'Attacker B',
        },
        tip: {
          format: {
            x: false,
            y: false,
            fill: false,
            'Attacker A': true,
            'Attacker B': true,
          },
        },
      }),
    ],
  });
}

function emptyPlot(message) {
  const wrap = document.createElement('div');
  wrap.className = 'placeholder';
  wrap.style.minHeight = '360px';
  wrap.style.display = 'grid';
  wrap.style.placeItems = 'center';
  wrap.textContent = message;
  return wrap;
}

function renderColorbar() {
  if (!rows.length) {
    document.getElementById('colorbar').innerHTML = '';
    return;
  }
  const alignment = currentAlignmentConfig();
  const alignmentValues = rows.map(alignmentValue);
  const alignmentMin = Math.min(...alignmentValues);
  const alignmentMax = Math.max(...alignmentValues);
  const w = 180, h = 12;
  document.getElementById('colorbar').innerHTML = '';
  const svg = d3.create('svg').attr('width', w + 80).attr('height', h + 28).style('vertical-align','middle');
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', `cosgrad-${currentDatasetName}`);
  const stops = 20;
  for (let i = 0; i <= stops; i++) {
    const t = i / stops;
    grad.append('stop').attr('offset', `${100*t}%`)
        .attr('stop-color', colorScale(alignmentMin + t * (alignmentMax - alignmentMin)));
  }
  svg.append('rect').attr('x', 0).attr('y', 4).attr('width', w).attr('height', h)
     .attr('fill', `url(#cosgrad-${currentDatasetName})`).attr('rx', 3);
  svg.append('text').attr('x', 0).attr('y', h + 22).attr('fill', '#5f6b7a').attr('font-size', 11)
     .text(alignmentMin.toFixed(2));
  svg.append('text').attr('x', w).attr('y', h + 22).attr('fill', '#5f6b7a').attr('font-size', 11)
     .attr('text-anchor', 'end').text(alignmentMax.toFixed(2));
  svg.append('text').attr('x', w + 8).attr('y', h).attr('fill', '#17202a').attr('font-size', 12)
     .text(alignment.axisLabel);
  document.getElementById('colorbar').append(svg.node());
}

// ---------- click-to-detail binding ----------
function bindPointClicks(svg) {
  const circles = svg.querySelectorAll('g[aria-label="dot"] circle');
  circles.forEach((node, i) => {
    node.style.cursor = 'pointer';
    node.addEventListener('click', () => {
      document.querySelectorAll('g[aria-label="dot"] circle').forEach(c => {
        c.setAttribute('stroke', defaultPointStroke());
        c.setAttribute('stroke-width', useDarkPointStroke ? '2.5' : '2');
      });
      node.setAttribute('stroke', '#17202a');
      node.setAttribute('stroke-width', '3');
      renderDetails(rows[i]);
    });
  });
}

function renderPlots() {
  renderAlignmentControls();
  if (!rows.length) {
    useDarkPointStroke = true;
    colorScale = () => '#9aa6b2';
    trend = { slope: 0, intercept: 0, r: null };
    document.getElementById('trend').innerHTML = '';
    document.getElementById('plot').innerHTML = '';
    document.getElementById('trend').append(emptyPlot('No rows available for this dataset yet.'));
    document.getElementById('plot').append(emptyPlot('No rows available for this dataset yet.'));
    renderColorbar();
    setPlotMode(currentPlotMode);
    return;
  }
  const alignmentValues = rows.map(alignmentValue);
  const alignmentMin = Math.min(...alignmentValues);
  const alignmentMax = Math.max(...alignmentValues);
  const alignmentSpan = alignmentMax - alignmentMin;
  useDarkPointStroke = rows.length <= 1 || !Number.isFinite(alignmentSpan) || Math.abs(alignmentSpan) < 1e-9;
  colorScale = d3.scaleSequential()
    .domain(useDarkPointStroke ? [alignmentMin - 0.01, alignmentMax + 0.01] : [alignmentMin, alignmentMax])
    .interpolator(t => d3.interpolateRdBu(t));
  trend = regression(rows, alignmentValue, d => d.Isum);

  document.getElementById('trend').innerHTML = '';
  document.getElementById('plot').innerHTML = '';
  const trendChart = buildTrendChart();
  const chart = buildQuadrantChart();
  document.getElementById('trend').append(trendChart);
  document.getElementById('plot').append(chart);
  if (rows.length) {
    bindPointClicks(trendChart);
    bindPointClicks(chart);
  }
  renderColorbar();
  setPlotMode(currentPlotMode);
}

function setPlotMode(mode) {
  currentPlotMode = mode;
  const copy = plotCopy();
  const showingTrend = mode === 'trend';
  document.getElementById('trend').classList.toggle('is-hidden', !showingTrend);
  document.getElementById('plot').classList.toggle('is-hidden', showingTrend);
  document.getElementById('quadrant-legend').classList.toggle('is-hidden', showingTrend);
  document.getElementById('trend-button').classList.toggle('active', showingTrend);
  document.getElementById('quadrant-button').classList.toggle('active', !showingTrend);
  document.getElementById('plot-title').textContent = copy[mode].title;
  document.getElementById('plot-subtitle').innerHTML = copy[mode].subtitle;
  document.getElementById('trend-stat').textContent = copy[mode].stat;
}

function setMainView(view) {
  const tableMode = view === 'table';
  const filterMode = view === 'filter';
  document.body.classList.toggle('table-mode', tableMode);
  document.body.classList.toggle('full-mode', tableMode || filterMode);
  document.getElementById('plot-card').classList.toggle('is-hidden', view !== 'plot');
  document.getElementById('table-card').classList.toggle('is-hidden', !tableMode);
  document.getElementById('filter-card').classList.toggle('is-hidden', !filterMode);
  document.getElementById('plot-view-button').classList.toggle('active', view === 'plot');
  document.getElementById('table-view-button').classList.toggle('active', tableMode);
  document.getElementById('filter-view-button').classList.toggle('active', filterMode);
}

document.getElementById('trend-button').addEventListener('click', () => setPlotMode('trend'));
document.getElementById('quadrant-button').addEventListener('click', () => setPlotMode('quadrant'));
document.getElementById('plot-view-button').addEventListener('click', () => setMainView('plot'));
document.getElementById('table-view-button').addEventListener('click', () => setMainView('table'));
document.getElementById('filter-view-button').addEventListener('click', () => setMainView('filter'));

function setMetric(metric) {
  if (metric === currentMetric) return;
  currentMetric = metric;
  document.getElementById('metric-success-button').classList.toggle('active', metric === 'success');
  document.getElementById('metric-confidence-button').classList.toggle('active', metric === 'confidence');
  allRows = prepareRows(DATASETS[currentDatasetName].rows);
  rows = motifFilteredRows();
  renderPlots();
  renderTable();
  renderFilteredTable();
  document.getElementById('placeholder').style.display = '';
  document.getElementById('detail').style.display = 'none';
}
document.getElementById('metric-success-button').addEventListener('click', () => setMetric('success'));
document.getElementById('metric-confidence-button').addEventListener('click', () => setMetric('confidence'));

function quadrant(ia, ib) {
  if (ia >= 0 && ib >= 0) return { tag: 'mutual cooperation', cls: 'q-coop' };
  if (ia <  0 && ib <  0) return { tag: 'mutual degradation', cls: 'q-suppress' };
  if (ia >= 0 && ib <  0) return { tag: 'A improves, B degrades', cls: 'q-asym-a' };
  return { tag: 'A degrades, B improves', cls: 'q-asym-b' };
}

function fmtPct(x, digits = 1) { return (x * 100).toFixed(digits) + '%'; }
function fx(x, d=3) { return x.toFixed(d); }

function plainChange(x) {
  return formatMetricValue(x);
}

function classForSigned(x) {
  return x >= 0 ? 'val-pos' : (x < 0 ? 'val-neg' : '');
}

function escapeHTML(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function motifDescription(motif) {
  return MOTIF_DESCRIPTIONS[motif] || '';
}

function motifDescriptionHtml(motifs) {
  const described = motifs
    .map(motif => ({
      label: MOTIF_LABELS[motif] || motif,
      text: motifDescription(motif),
    }))
    .filter(item => item.text);
  if (!described.length) return '';
  return described
    .map(item => `<div><strong>${escapeHTML(item.label)}:</strong> ${escapeHTML(item.text)}</div>`)
    .join('');
}

const tableColumns = [
  { key: 'anchor', label: 'A source', value: d => d.anchor, format: d => escapeHTML(d.anchor) },
  { key: 'partner', label: 'B source', value: d => d.partner, format: d => escapeHTML(d.partner) },
  { key: 'target_class', label: 'Target', value: d => d.target_class, format: d => escapeHTML(d.target_class) },
  { key: 'motif_label', label: 'Motif', value: d => d.motif_label || '', format: d => escapeHTML(MOTIF_LABELS[d.motif_label] || d.motif_label || 'C1 layered') },
  { key: 'Ia', label: 'A change', value: d => d.Ia, numeric: true, format: d => `<span class="${classForSigned(d.Ia)}">${plainChange(d.Ia)}</span>` },
  { key: 'Ib', label: 'B change', value: d => d.Ib, numeric: true, format: d => `<span class="${classForSigned(d.Ib)}">${plainChange(d.Ib)}</span>` },
  { key: 'Imean', label: 'Mean A/B change', value: d => d.Imean, numeric: true, format: d => `<span class="${classForSigned(d.Imean)}">${plainChange(d.Imean)}</span>` },
  { key: 'Iworst', label: 'Worst single change', value: d => d.Iworst, numeric: true, format: d => `<span class="${classForSigned(d.Iworst)}">${plainChange(d.Iworst)}</span>` },
  { key: 'cos_attack', label: () => currentDatasetName === 'C6' ? 'Target-pair distance' : 'Attack alignment', value: d => d.cos_attack, numeric: true, format: d => fx(d.cos_attack) },
  { key: 'd_ab', label: () => currentDatasetName === 'C6' ? 'Target image distance' : 'Source distance', value: d => d.d_ab, numeric: true, format: d => fx(d.d_ab) },
];

let tableSort = { key: 'Ia', dir: 'asc' };
const columnVisibilityByState = {};

function columnStateKey() {
  return `${currentDatasetName}:${currentMetric}`;
}

function defaultColumnVisible(column) {
  if (column.key === 'motif_label') return isMotifDataset();
  if (column.key === 'Iworst') return false;
  return true;
}

function columnVisibility() {
  const stateKey = columnStateKey();
  if (!columnVisibilityByState[stateKey]) {
    columnVisibilityByState[stateKey] = Object.fromEntries(
      tableColumns.map(column => [column.key, defaultColumnVisible(column)])
    );
  }
  tableColumns.forEach(column => {
    if (!(column.key in columnVisibilityByState[stateKey])) {
      columnVisibilityByState[stateKey][column.key] = defaultColumnVisible(column);
    }
  });
  return columnVisibilityByState[stateKey];
}

function visibleTableColumns() {
  const visibility = columnVisibility();
  const visible = tableColumns.filter(column => visibility[column.key]);
  return visible.length ? visible : tableColumns.slice(0, 1);
}

function ensureVisibleSortColumn() {
  const visibleKeys = new Set(visibleTableColumns().map(column => column.key));
  if (!visibleKeys.has(tableSort.key)) {
    const fallback = visibleTableColumns().find(column => column.numeric) || visibleTableColumns()[0];
    tableSort = { key: fallback.key, dir: fallback.numeric ? 'asc' : 'asc' };
  }
}

function toggleColumn(key, checked) {
  const visibility = columnVisibility();
  visibility[key] = checked;
  if (!tableColumns.some(column => visibility[column.key])) {
    visibility[key] = true;
  }
  ensureVisibleSortColumn();
  renderTable();
  renderFilteredTable();
}

function renderColumnControls() {
  const visibility = columnVisibility();
  const controlsHtml = tableColumns.map(column => {
    const checked = visibility[column.key] ? 'checked' : '';
    return `<label class="column-option"><input type="checkbox" data-column="${column.key}" ${checked}>${column.label}</label>`;
  }).join('');
  ['table-column-controls', 'filter-column-controls'].forEach(id => {
    document.getElementById(id).innerHTML = controlsHtml;
  });
  document.querySelectorAll('.column-option input').forEach(input => {
    input.addEventListener('change', () => toggleColumn(input.dataset.column, input.checked));
  });
}

const presetSorts = {
  anchorDrop: { key: 'Ia', dir: 'asc' },
  partnerDrop: { key: 'Ib', dir: 'asc' },
  meanDrop: { key: 'Imean', dir: 'asc' },
  alignmentDesc: { key: 'cos_attack', dir: 'desc' },
  alignmentAsc: { key: 'cos_attack', dir: 'asc' },
};

function sortRows(data) {
  ensureVisibleSortColumn();
  const column = tableColumns.find(c => c.key === tableSort.key);
  const sign = tableSort.dir === 'asc' ? 1 : -1;
  return [...data].sort((a, b) => {
    const av = column.value(a);
    const bv = column.value(b);
    if (typeof av === 'string') return sign * av.localeCompare(bv);
    return sign * (av - bv);
  });
}

function renderRowsTable(containerId, data, rerender) {
  ensureVisibleSortColumn();
  const columns = visibleTableColumns();
  const sorted = sortRows(data);
  const headers = columns.map(col => {
    const mark = col.key === tableSort.key ? (tableSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
    const label = typeof col.label === 'function' ? col.label() : col.label;
    return `<th data-key="${col.key}" class="${col.numeric ? 'num' : ''}">${label}${mark}</th>`;
  }).join('');
  const body = sorted.map(row => {
    const cells = columns.map(col => {
      const cls = col.numeric ? ' class="num"' : '';
      return `<td${cls}>${col.format(row)}</td>`;
    }).join('');
    return `<tr data-row-id="${escapeHTML(row.row_id)}">${cells}</tr>`;
  }).join('');

  document.getElementById(containerId).innerHTML = `<table>
    <thead><tr>${headers}</tr></thead>
    <tbody>${body}</tbody>
  </table>`;

  document.querySelectorAll(`#${containerId} th`).forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      tableSort = {
        key,
        dir: tableSort.key === key && tableSort.dir === 'asc' ? 'desc' : 'asc',
      };
      rerender();
    });
  });
  document.querySelectorAll(`#${containerId} tbody tr`).forEach(tr => {
    tr.addEventListener('click', () => {
      const row = rows.find(d => d.row_id === tr.dataset.rowId);
      if (row) renderDetails(row);
    });
  });
}

function renderTable() {
  renderColumnControls();
  document.getElementById('table-unit').textContent =
    `${metricShortLabel()}; change columns use ${metricUnit()}.`;
  renderRowsTable('table-wrap', rows, renderTable);
}

function motifFilteredRows() {
  if (!isMotifDataset() || activeMotifs.size === 0) return allRows;
  return allRows.filter(row => activeMotifs.has(row.motif_label));
}

function renderMotifFilter() {
  const controls = document.getElementById('motif-controls');
  controls.classList.toggle('is-hidden', !isMotifDataset());
  if (!isMotifDataset()) {
    document.getElementById('motif-filter').innerHTML = '';
    document.getElementById('motif-description').textContent = '';
    return;
  }
  document.getElementById('motif-filter-label').textContent = `${DATASETS[currentDatasetName].label} motif filter`;
  const motifs = [...new Set(allRows.map(row => row.motif_label).filter(Boolean))].sort();
  const motifCounts = d3.rollup(allRows, values => values.length, row => row.motif_label);
  document.getElementById('motif-filter').innerHTML = motifs.map(motif => {
    const active = activeMotifs.has(motif);
    const count = motifCounts.get(motif) || 0;
    return `<button class="filter-chip ${active ? 'active' : ''}" type="button" data-motif="${escapeHTML(motif)}">${escapeHTML(MOTIF_LABELS[motif] || motif)} (${count})</button>`;
  }).join('');

  document.querySelectorAll('#motif-filter .filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const motif = chip.dataset.motif;
      if (activeMotifs.has(motif)) {
        activeMotifs.delete(motif);
      } else {
        activeMotifs.add(motif);
      }
      rows = motifFilteredRows();
      renderMotifFilter();
      renderPlots();
      renderTable();
      renderFilteredTable();
    });
  });
  const description = document.getElementById('motif-description');
  if (currentDatasetName === 'C2mini' || currentDatasetName === 'C6') {
    const selectedMotifs = activeMotifs.size ? motifs.filter(motif => activeMotifs.has(motif)) : motifs;
    description.innerHTML = motifDescriptionHtml(selectedMotifs);
  } else {
    description.textContent = activeMotifs.size === 0
      ? `No motif selected: showing all ${DATASETS[currentDatasetName].label} motifs.`
      : `${activeMotifs.size} motif${activeMotifs.size === 1 ? '' : 's'} selected.`;
  }
}

const filterState = {
  anchor: new Set(),
  partner: new Set(),
  target_class: new Set(),
};

const filterConfig = [
  { key: 'anchor', elementId: 'filter-anchor' },
  { key: 'partner', elementId: 'filter-partner' },
  { key: 'target_class', elementId: 'filter-target' },
];

function rowMatchesFilter(row, key) {
  const selected = filterState[key];
  if (selected.size === 0) return true;
  if (key === 'target_class') {
    return row.target_classes.some(value => selected.has(value));
  }
  return selected.has(row[key]);
}

function rowsMatchingFilters({ skipKey = null } = {}) {
  return rows.filter(row => filterConfig.every(({ key }) => {
    if (key === skipKey) return true;
    return rowMatchesFilter(row, key);
  }));
}

function valuesFromRows(data, key) {
  const values = key === 'target_class'
    ? data.flatMap(row => row.target_classes)
    : data.map(row => row[key]);
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b)));
}

function availableValues(key) {
  return valuesFromRows(rowsMatchingFilters({ skipKey: key }), key);
}

function pruneUnavailableFilters() {
  let changed = true;
  while (changed) {
    changed = false;
    filterConfig.forEach(({ key }) => {
      const available = new Set(availableValues(key));
      [...filterState[key]].forEach(value => {
        if (!available.has(value)) {
          filterState[key].delete(value);
          changed = true;
        }
      });
    });
  }
}

function filteredRows() {
  return rowsMatchingFilters();
}

function renderFilterChips() {
  pruneUnavailableFilters();
  filterConfig.forEach(({ key, elementId }) => {
    const chips = availableValues(key).sort((a, b) => {
      const aActive = filterState[key].has(a);
      const bActive = filterState[key].has(b);
      if (aActive !== bActive) return aActive ? -1 : 1;
      return String(a).localeCompare(String(b));
    }).map(value => {
      const active = filterState[key].has(value);
      return `<button class="filter-chip ${active ? 'active' : ''}" type="button" data-key="${key}" data-value="${escapeHTML(value)}">${escapeHTML(value)}</button>`;
    }).join('');
    document.getElementById(elementId).innerHTML = chips;
  });

  document.querySelectorAll('#filter-anchor .filter-chip, #filter-partner .filter-chip, #filter-target .filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const key = chip.dataset.key;
      const value = chip.dataset.value;
      if (filterState[key].has(value)) {
        filterState[key].delete(value);
      } else {
        filterState[key].add(value);
      }
      renderFilteredTable();
    });
  });
}

function renderFilteredTable() {
  pruneUnavailableFilters();
  renderColumnControls();
  renderFilterChips();
  const data = filteredRows();
  document.getElementById('filter-summary').textContent =
    `${data.length} of ${rows.length} cells shown`;
  document.getElementById('filter-table-unit').textContent =
    `${metricShortLabel()}; change columns use ${metricUnit()}.`;
  renderRowsTable('filter-table-wrap', data, renderFilteredTable);
}

document.getElementById('clear-filters').addEventListener('click', () => {
  Object.values(filterState).forEach(selected => selected.clear());
  renderFilteredTable();
});

function selectDataset(name) {
  currentDatasetName = name;
  allRows = prepareRows(DATASETS[name].rows);
  activeMotifs.clear();
  rows = motifFilteredRows();
  Object.values(filterState).forEach(selected => selected.clear());
  document.getElementById('dataset-c1-button').classList.toggle('active', name === 'C1');
  document.getElementById('dataset-c2-button').classList.toggle('active', name === 'C2');
  document.getElementById('dataset-c2-mini-button').classList.toggle('active', name === 'C2mini');
  document.getElementById('dataset-c6-button').classList.toggle('active', name === 'C6');
  renderMotifFilter();
  renderPlots();
  renderTable();
  renderFilteredTable();
}

document.getElementById('dataset-c1-button').addEventListener('click', () => selectDataset('C1'));
document.getElementById('dataset-c2-button').addEventListener('click', () => selectDataset('C2'));
document.getElementById('dataset-c2-mini-button').addEventListener('click', () => selectDataset('C2mini'));
document.getElementById('dataset-c6-button').addEventListener('click', () => selectDataset('C6'));

document.getElementById('sort-select').addEventListener('change', ev => {
  tableSort = presetSorts[ev.target.value];
  renderTable();
});

selectDataset('C1');
setMainView('plot');

function pill(value, {digits = metricDigits(), pm = null, unit = ''} = {}) {
  const cls = value > 0 ? 'pos' : (value < 0 ? 'neg' : '');
  const sign = value >= 0 ? '+' : '';
  const main = `${sign}${value.toFixed(digits)}${unit ? '\u202f' + unit : ''}`;
  const pmHtml = pm == null ? '' : ` <span class="pm">±\u202f${pm.toFixed(digits)}</span>`;
  return `<span class="value-pill ${cls}">${main}${pmHtml}</span>`;
}

function soloValue(value) {
  if (currentMetric === 'confidence') return value.toFixed(3);
  return fmtPct(value);
}

function alternateMetricChange(d, attacker) {
  if (currentMetric === 'confidence') {
    return d[`I_${attacker}_s`] * 100;
  }
  return d[`I_${attacker}_c`];
}

function alternateMetricDigits() {
  return currentMetric === 'confidence' ? 1 : 3;
}

function alternateMetricUnit() {
  return currentMetric === 'confidence' ? 'percentage points' : 'raw confidence score';
}

function renderDetails(d) {
  document.getElementById('placeholder').style.display = 'none';
  const q = quadrant(d.Ia, d.Ib);
  const det = document.getElementById('detail');
  const c2Like = isC2LikeDataset();
  const c6Like = currentDatasetName === 'C6';
  const targetSpecific = c2Like || (
    d.target_a_class_name && d.target_b_class_name && d.target_a_class_name !== d.target_b_class_name
  ) || c6Like;
  const hasCrossGeometry = targetSpecific && d.a_cross != null && d.b_cross != null && d.target_target_distance != null;
  const hasTargetPairGeometry = c6Like && d.target_target_distance != null;
  const reps = d.n == null ? currentDatasetName : `n = ${d.n} reps`;
  const metricLabel = currentMetric === 'confidence' ? 'target confidence' : 'success rate';
  const combinedLabel = d.Isum >= 0 ? 'above solo' : 'below solo';
  const solo_a = currentMetric === 'confidence' ? d.solo_anc_c : d.solo_anc_s;
  const solo_b = currentMetric === 'confidence' ? d.solo_par_c : d.solo_par_s;
  const dual_a = currentMetric === 'confidence' ? d.dual_anc_c : d.dual_anc_s;
  const dual_b = currentMetric === 'confidence' ? d.dual_par_c : d.dual_par_s;
  const scenarioText = motifDescription(d.motif_label);
  const scenarioBlock = scenarioText ? `
    <div class="detail-card">
      <div class="block-label">Scenario</div>
      <div class="detail-summary">${escapeHTML(scenarioText)}</div>
    </div>
  ` : '';

  const soloBlock = solo_a == null ? (d.motif_label ? `
    <div class="detail-card">
      <div class="block-label">Motif</div>
      <div class="kv">
        <div class="k">Label</div><div class="v"><span class="value-pill">${escapeHTML(d.motif_label)}</span></div>
      </div>
    </div>
  ` : '') : `
    <div class="detail-card">
      <div class="block-label">Solo → dual (${metricLabel})</div>
      <div class="formula">\\(I_A = \\overline{x_A^{\\text{dual}}} - \\overline{x_A^{\\text{solo}}}\\), where \\(x\\) is the per-rep ${metricLabel}.</div>
      <div class="kv">
        <div class="k">\\(A\\)</div><div class="v"><span class="value-pill">${soloValue(solo_a)}</span><span class="arrow">→</span><span class="value-pill">${soloValue(dual_a)}</span></div>
        <div class="k">\\(B\\)</div><div class="v"><span class="value-pill">${soloValue(solo_b)}</span><span class="arrow">→</span><span class="value-pill">${soloValue(dual_b)}</span></div>
      </div>
    </div>
  `;

  const otherMetric = currentMetric === 'confidence' ? 'success rate' : 'target confidence';
  const otherIa = alternateMetricChange(d, 'a');
  const otherIb = alternateMetricChange(d, 'b');
  const otherBlock = `
    <div class="detail-card">
      <div class="block-label">For comparison: ${otherMetric}</div>
      <div class="formula">Unit: ${alternateMetricUnit()}.</div>
      <div class="kv">
        <div class="k">\\(I_A\\)</div><div class="v">${pill(otherIa, {digits: alternateMetricDigits()})}</div>
        <div class="k">\\(I_B\\)</div><div class="v">${pill(otherIb, {digits: alternateMetricDigits()})}</div>
      </div>
    </div>
  `;

  det.style.display = 'block';
  const targetA = escapeHTML(d.target_a_class_name || d.target_class || 'airplane');
  const targetB = escapeHTML(d.target_b_class_name || d.target_class || 'airplane');
  det.innerHTML = `
    <div class="attacker-stack">
      <div class="attacker-route">
        <span class="attacker-route-main">
          <span class="attacker-name">${escapeHTML(d.anchor)}</span>
          <span class="attacker-arrow">-&gt;</span>
          <span class="attacker-target">${targetA}</span>
        </span>
        <span class="attacker-role">A</span>
      </div>
      <div class="attacker-route">
        <span class="attacker-route-main">
          <span class="attacker-name">${escapeHTML(d.partner)}</span>
          <span class="attacker-arrow">-&gt;</span>
          <span class="attacker-target">${targetB}</span>
        </span>
        <span class="attacker-role">B</span>
      </div>
    </div>
    <div class="role-sub">
      <span class="qtag ${q.cls}">${q.tag}</span>
      <span class="n-chip">${reps}</span>
    </div>
    <div class="detail-summary">
      Combined shift is <strong class="${classForSigned(d.Isum)}">${formatMetricValue(d.Isum)}</strong>
      ${metricUnit()} ${combinedLabel}.
    </div>

    <div class="detail-card">
      <div class="block-label">Interaction (${metricLabel})</div>
      <div class="formula">\\(I_A\\), \\(I_B\\) = change in ${metricLabel} when each attacker runs alongside its partner vs. alone. Unit: ${metricUnit()}.</div>
      <div class="kv">
        <div class="k">\\(I_A\\)</div><div class="v">${pill(d.Ia, {pm: d.Isem_a})}</div>
        <div class="k">\\(I_B\\)</div><div class="v">${pill(d.Ib, {pm: d.Isem_b})}</div>
        <div class="k">\\(I_A + I_B\\)</div><div class="v">${pill(d.Isum)}</div>
      </div>
    </div>

    ${soloBlock}
    ${scenarioBlock}

    <div class="detail-card">
      <div class="block-label">Geometry (cosine distances)</div>
      <div class="formula">\\(d(x, y) = 1 - \\cos(\\phi_x, \\phi_y)\\) between penultimate-layer features, so \\(\\cos(A, B) = 1 - d(A, B)\\).</div>
      ${hasTargetPairGeometry
        ? `<div class="formula">For C6, A and B are two target images from the same true class. The alignment diagnostic is their clean-representation distance; smaller values mean the target images are closer before poisoning.</div>`
        : !targetSpecific
        ? `<div class="formula">For C1 attack-direction alignment: \\(\\vec{\\delta}_A = \\mathrm{normalize}(c_T - c_A)\\), \\(\\vec{\\delta}_B = \\mathrm{normalize}(c_T - c_B)\\), and \\(\\cos(\\vec{\\delta}_A, \\vec{\\delta}_B) = \\vec{\\delta}_A \\cdot \\vec{\\delta}_B\\).</div>`
        : `<div class="formula">For target-specific attack-direction alignment: \\(\\vec{\\delta}_A = \\mathrm{normalize}(c_{T_A} - c_A)\\), \\(\\vec{\\delta}_B = \\mathrm{normalize}(c_{T_B} - c_B)\\), and \\(\\cos(\\vec{\\delta}_A, \\vec{\\delta}_B) = \\vec{\\delta}_A \\cdot \\vec{\\delta}_B\\). ${hasCrossGeometry ? 'Cross-alignment gap = \\([d(A, T_B) + d(B, T_A)] - [d(A, T_A) + d(B, T_B)]\\).' : ''}</div>`}
      <div class="kv">
        <div class="k">${hasTargetPairGeometry ? 'Target image distance' : '\\(d(A, B)\\)'}</div><div class="v"><span class="value-pill">${fx(d.d_ab)}</span></div>
        <div class="k">${!targetSpecific ? '\\(d(A, T)\\)' : '\\(d(A, T_A)\\)'}</div><div class="v"><span class="value-pill">${fx(d.d_anc_tgt)}</span></div>
        <div class="k">${!targetSpecific ? '\\(d(B, T)\\)' : '\\(d(B, T_B)\\)'}</div><div class="v"><span class="value-pill">${fx(d.d_par_tgt)}</span></div>
        ${hasCrossGeometry
          ? `<div class="k">\\(d(A, T_B)\\)</div><div class="v"><span class="value-pill">${fx(d.a_cross)}</span></div>
        <div class="k">\\(d(B, T_A)\\)</div><div class="v"><span class="value-pill">${fx(d.b_cross)}</span></div>
        <div class="k">\\(d(T_A, T_B)\\)</div><div class="v"><span class="value-pill">${fx(d.target_target_distance)}</span></div>
        <div class="k">Cross gap</div><div class="v"><span class="value-pill">${fx(d.cross_alignment_gap)}</span></div>`
          : ''}
        <div class="k">\\(\\cos(A, B)\\)</div><div class="v"><span class="value-pill">${fx(d.cos_source)}</span></div>
        <div class="k">${hasTargetPairGeometry ? 'Gradient cosine' : '\\(\\cos(\\vec{\\delta}_A, \\vec{\\delta}_B)\\)'}</div><div class="v"><span class="value-pill">${fx(hasTargetPairGeometry ? d.gradient_cosine : (targetSpecific ? (d.cos_attack_dir ?? d.cos_attack) : d.cos_attack))}</span></div>
      </div>
    </div>

    ${otherBlock}
  `;
  if (window.renderMathInElement) {
    renderMathInElement(det, {
      delimiters: [
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false},
      ],
      throwOnError: false,
    });
  }
}
</script>
</body>
</html>
"""


def _read_csv_rows(path):
    with Path(path).open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def _sem_values(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / (len(values) ** 0.5)) if len(values) > 1 else 0.0


def _float_or(value, default):
    if value in (None, ''):
        return default
    return float(value)


def _solo_means_by_attacker(root):
    solo_means = {}
    for path in sorted((root / 'solo').glob('*.csv')):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        attacker_id = rows[0]['attacker_id']
        solo_means[attacker_id] = dict(
            success=_mean([row['success'] for row in rows]),
            confidence=_mean([row['adv_confidence'] for row in rows]),
        )
    return solo_means


def _c2_mini_motif(condition):
    mapping = {
        'dog_vs_frog_easy_close': 'c2_mini_easy_close',
        'dog_vs_frog_easy_far': 'c2_mini_easy_far',
        'dog_vs_frog_hard_close': 'c2_mini_hard_close',
        'dog_vs_frog_hard_far': 'c2_mini_hard_far',
    }
    if condition.endswith('__clone'):
        return 'same_source_clone'
    return mapping.get(condition, condition)


def _dog_cat_vs_truck_auto_override():
    solo_means = _solo_means_by_attacker(DOG_CAT_TRUCK_ROOT)

    dog_adv, dog_success, truck_adv, truck_success = [], [], [], []
    for path in sorted((DOG_CAT_TRUCK_ROOT / 'dual').glob('*.csv')):
        rows = _read_csv_rows(path)
        by_attacker = {}
        for row in rows:
            by_attacker.setdefault(row['attacker_id'], []).append(row)

        per_source = {}
        for attacker_id, attacker_rows in by_attacker.items():
            if attacker_id not in solo_means:
                raise KeyError(f'Missing solo output for {attacker_id}.')
            source_name = attacker_rows[0]['source_class_name']
            per_source[source_name] = dict(
                adv=_mean([row['adv_confidence'] for row in attacker_rows]) - solo_means[attacker_id]['confidence'],
                success=_mean([row['success'] for row in attacker_rows]) - solo_means[attacker_id]['success'],
            )

        dog_adv.append(per_source['dog']['adv'])
        dog_success.append(per_source['dog']['success'])
        truck_adv.append(per_source['truck']['adv'])
        truck_success.append(per_source['truck']['success'])

    if not dog_adv:
        raise ValueError(f'No dual CSVs found in {DOG_CAT_TRUCK_ROOT / "dual"}.')

    return dict(
        n=len(dog_adv),
        I_a_s=_mean(dog_success),
        I_b_s=_mean(truck_success),
        I_a_s_sem=_sem_values(dog_success),
        I_b_s_sem=_sem_values(truck_success),
        I_a_c=_mean(dog_adv),
        I_b_c=_mean(truck_adv),
        I_a_c_sem=_sem_values(dog_adv),
        I_b_c_sem=_sem_values(truck_adv),
    )


def _same_source_clone_rows():
    solo_means = _solo_means_by_attacker(C2_MINI_ROOT)

    by_condition = {}
    for path in sorted((C2_MINI_ROOT / 'dual').glob('dual_*__clone_rep*.csv')):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        condition = rows[0]['condition']
        ordered_attacker_ids = list(dict.fromkeys(row['attacker_id'] for row in rows))
        if len(ordered_attacker_ids) != 2:
            raise ValueError(f'Expected two attackers in {path}.')

        per_attacker = []
        for attacker_id in ordered_attacker_ids:
            if attacker_id not in solo_means:
                raise KeyError(f'Missing solo output for {attacker_id}.')
            attacker_rows = [row for row in rows if row['attacker_id'] == attacker_id]
            per_attacker.append(dict(
                source=attacker_rows[0]['source_class_name'],
                target=attacker_rows[0]['target_adv_class_name'],
                source_target_distance=float(attacker_rows[0]['source_target_distance']),
                solo_s=solo_means[attacker_id]['success'],
                solo_c=solo_means[attacker_id]['confidence'],
                dual_s=_mean([row['success'] for row in attacker_rows]),
                dual_c=_mean([row['adv_confidence'] for row in attacker_rows]),
            ))
        by_condition.setdefault(condition, []).append(per_attacker)

    payload = []
    for condition, reps in sorted(by_condition.items()):
        first = reps[0][0]
        source = first['source']
        target = first['target']
        a_success = [rep[0]['dual_s'] - rep[0]['solo_s'] for rep in reps]
        b_success = [rep[1]['dual_s'] - rep[1]['solo_s'] for rep in reps]
        a_confidence = [rep[0]['dual_c'] - rep[0]['solo_c'] for rep in reps]
        b_confidence = [rep[1]['dual_c'] - rep[1]['solo_c'] for rep in reps]

        payload.append(dict(
            anchor=source,
            partner=source,
            n=len(reps),
            d_ab=0.0,
            d_anc_tgt=first['source_target_distance'],
            d_par_tgt=first['source_target_distance'],
            I_a_s=_mean(a_success),
            I_a_s_sem=_sem_values(a_success),
            I_b_s=_mean(b_success),
            I_b_s_sem=_sem_values(b_success),
            I_a_c=_mean(a_confidence),
            I_a_c_sem=_sem_values(a_confidence),
            I_b_c=_mean(b_confidence),
            I_b_c_sem=_sem_values(b_confidence),
            solo_anc_s=_mean([rep[0]['solo_s'] for rep in reps]),
            solo_par_s=_mean([rep[1]['solo_s'] for rep in reps]),
            solo_anc_c=_mean([rep[0]['solo_c'] for rep in reps]),
            solo_par_c=_mean([rep[1]['solo_c'] for rep in reps]),
            dual_anc_s=_mean([rep[0]['dual_s'] for rep in reps]),
            dual_par_s=_mean([rep[1]['dual_s'] for rep in reps]),
            dual_anc_c=_mean([rep[0]['dual_c'] for rep in reps]),
            dual_par_c=_mean([rep[1]['dual_c'] for rep in reps]),
            cos_attack=1.0,
            anchor_role='same-source',
            cos_source=1.0,
            target_class=target,
            target_classes=[target],
            target_a_class_name=target,
            target_b_class_name=target,
            same_source_clone=True,
            condition=condition,
        ))
    return payload


def _fork_far_apart_rows(attack_direction_cos):
    solo_means = _solo_means_by_attacker(FORK_FAR_APART_ROOT)

    by_condition = {}
    for path in sorted((FORK_FAR_APART_ROOT / 'dual').glob('*.csv')):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        condition = rows[0]['condition'] or rows[0]['pairing_id'] or path.stem
        ordered_attacker_ids = list(dict.fromkeys(row['attacker_id'] for row in rows))
        if len(ordered_attacker_ids) != 2:
            raise ValueError(f'Expected two attackers in {path}.')

        per_attacker = []
        for attacker_id in ordered_attacker_ids:
            if attacker_id not in solo_means:
                raise KeyError(f'Missing solo output for {attacker_id}.')
            attacker_rows = [row for row in rows if row['attacker_id'] == attacker_id]
            per_attacker.append(dict(
                source=attacker_rows[0]['source_class_name'],
                target=attacker_rows[0]['target_adv_class_name'],
                source_target_distance=float(attacker_rows[0]['source_target_distance']),
                solo_s=solo_means[attacker_id]['success'],
                solo_c=solo_means[attacker_id]['confidence'],
                dual_s=_mean([row['success'] for row in attacker_rows]),
                dual_c=_mean([row['adv_confidence'] for row in attacker_rows]),
            ))
        by_condition.setdefault(condition, []).append(dict(meta=rows[0], attackers=per_attacker))

    payload = []
    for index, (condition, reps) in enumerate(sorted(by_condition.items())):
        first_meta = reps[0]['meta']
        first_a, first_b = reps[0]['attackers']
        target_a = first_a['target']
        target_b = first_b['target']
        target_classes = list(dict.fromkeys([target_a, target_b]))
        target_label = target_a if target_a == target_b else f'{target_a} / {target_b}'

        a_success = [rep['attackers'][0]['dual_s'] - rep['attackers'][0]['solo_s'] for rep in reps]
        b_success = [rep['attackers'][1]['dual_s'] - rep['attackers'][1]['solo_s'] for rep in reps]
        a_confidence = [rep['attackers'][0]['dual_c'] - rep['attackers'][0]['solo_c'] for rep in reps]
        b_confidence = [rep['attackers'][1]['dual_c'] - rep['attackers'][1]['solo_c'] for rep in reps]

        d_ab = _float_or(first_meta['source_source_distance'], 0.0)
        a_self = _float_or(first_meta['a_self'], first_a['source_target_distance'])
        b_self = _float_or(first_meta['b_self'], first_b['source_target_distance'])
        a_cross = _float_or(first_meta['a_cross'], b_self)
        b_cross = _float_or(first_meta['b_cross'], a_self)
        target_target_distance = _float_or(first_meta['target_target_distance'], 0.0)
        cross_alignment_gap = _float_or(first_meta['cross_alignment_gap'], (a_cross + b_cross) - (a_self + b_self))

        cos_attack = attack_direction_cos(first_a['source'], target_a, first_b['source'], target_b)
        payload.append(dict(
            row_id=f'C1-fork-{index}',
            dataset='C1',
            anchor=first_a['source'],
            partner=first_b['source'],
            target_class=target_label,
            target_classes=target_classes,
            target_a_class_name=target_a,
            target_b_class_name=target_b,
            n=len(reps),
            I_a_s=_mean(a_success),
            I_b_s=_mean(b_success),
            I_a_s_sem=_sem_values(a_success),
            I_b_s_sem=_sem_values(b_success),
            I_a_c=_mean(a_confidence),
            I_b_c=_mean(b_confidence),
            I_a_c_sem=_sem_values(a_confidence),
            I_b_c_sem=_sem_values(b_confidence),
            solo_anc_s=_mean([rep['attackers'][0]['solo_s'] for rep in reps]),
            solo_par_s=_mean([rep['attackers'][1]['solo_s'] for rep in reps]),
            dual_anc_s=_mean([rep['attackers'][0]['dual_s'] for rep in reps]),
            dual_par_s=_mean([rep['attackers'][1]['dual_s'] for rep in reps]),
            solo_anc_c=_mean([rep['attackers'][0]['solo_c'] for rep in reps]),
            solo_par_c=_mean([rep['attackers'][1]['solo_c'] for rep in reps]),
            dual_anc_c=_mean([rep['attackers'][0]['dual_c'] for rep in reps]),
            dual_par_c=_mean([rep['attackers'][1]['dual_c'] for rep in reps]),
            d_ab=d_ab,
            cos_source=1.0 - d_ab,
            d_anc_tgt=a_self,
            d_par_tgt=b_self,
            a_cross=a_cross,
            b_cross=b_cross,
            target_target_distance=target_target_distance,
            cross_alignment_gap=cross_alignment_gap,
            cos_attack=cos_attack,
            cos_attack_dir=cos_attack,
            motif_label='fork_far_apart',
            source_stratum=first_meta['source_stratum'] or 'same-source',
            alignment_type=first_meta['alignment_type'] or 'same_source_fork',
            condition=condition,
            same_source_fork=True,
        ))
    return payload


def _c2_mini_rows(attack_direction_cos):
    solo_means = _solo_means_by_attacker(C2_MINI_ROOT)
    by_condition = {}
    for path in sorted((C2_MINI_ROOT / 'dual').glob('*.csv')):
        rows = _read_csv_rows(path)
        if not rows:
            continue

        condition = rows[0]['condition']
        if condition.endswith('__clone'):
            continue
        ordered_attacker_ids = list(dict.fromkeys(row['attacker_id'] for row in rows))
        if len(ordered_attacker_ids) != 2:
            raise ValueError(f'Expected two attackers in {path}.')

        per_attacker = []
        for attacker_id in ordered_attacker_ids:
            if attacker_id not in solo_means:
                raise KeyError(f'Missing solo output for {attacker_id}.')
            attacker_rows = [row for row in rows if row['attacker_id'] == attacker_id]
            per_attacker.append(dict(
                source=attacker_rows[0]['source_class_name'],
                target=attacker_rows[0]['target_adv_class_name'],
                source_target_distance=float(attacker_rows[0]['source_target_distance']),
                solo_s=solo_means[attacker_id]['success'],
                solo_c=solo_means[attacker_id]['confidence'],
                dual_s=_mean([row['success'] for row in attacker_rows]),
                dual_c=_mean([row['adv_confidence'] for row in attacker_rows]),
            ))
        by_condition.setdefault(condition, []).append(dict(meta=rows[0], attackers=per_attacker))

    payload = []
    for index, (condition, reps) in enumerate(sorted(by_condition.items())):
        first_meta = reps[0]['meta']
        first_a, first_b = reps[0]['attackers']
        target_a = first_a['target']
        target_b = first_b['target']
        target_classes = list(dict.fromkeys([target_a, target_b]))
        target_label = target_a if target_a == target_b else f'{target_a} / {target_b}'

        a_success = [rep['attackers'][0]['dual_s'] - rep['attackers'][0]['solo_s'] for rep in reps]
        b_success = [rep['attackers'][1]['dual_s'] - rep['attackers'][1]['solo_s'] for rep in reps]
        a_confidence = [rep['attackers'][0]['dual_c'] - rep['attackers'][0]['solo_c'] for rep in reps]
        b_confidence = [rep['attackers'][1]['dual_c'] - rep['attackers'][1]['solo_c'] for rep in reps]

        d_ab = _float_or(first_meta['source_source_distance'], 0.0)
        a_self = _float_or(first_meta['a_self'], first_a['source_target_distance'])
        b_self = _float_or(first_meta['b_self'], first_b['source_target_distance'])
        a_cross = _float_or(first_meta['a_cross'], a_self)
        b_cross = _float_or(first_meta['b_cross'], b_self)
        target_target_distance = _float_or(first_meta['target_target_distance'], 0.0)
        cross_alignment_gap = _float_or(first_meta['cross_alignment_gap'], (a_cross + b_cross) - (a_self + b_self))
        alignment_type = first_meta['alignment_type'] or ('same_source_clone' if condition.endswith('__clone') else 'mixed')
        source_stratum = first_meta['source_stratum'] or ('same-source' if condition.endswith('__clone') else '')

        payload.append(dict(
            row_id=f'C2mini-{index}',
            dataset='C2mini',
            anchor=first_a['source'],
            partner=first_b['source'],
            target_class=target_label,
            target_classes=target_classes,
            target_a_class_name=target_a,
            target_b_class_name=target_b,
            n=len(reps),
            I_a_s=_mean(a_success),
            I_b_s=_mean(b_success),
            I_a_s_sem=_sem_values(a_success),
            I_b_s_sem=_sem_values(b_success),
            I_a_c=_mean(a_confidence),
            I_b_c=_mean(b_confidence),
            I_a_c_sem=_sem_values(a_confidence),
            I_b_c_sem=_sem_values(b_confidence),
            solo_anc_s=_mean([rep['attackers'][0]['solo_s'] for rep in reps]),
            solo_par_s=_mean([rep['attackers'][1]['solo_s'] for rep in reps]),
            dual_anc_s=_mean([rep['attackers'][0]['dual_s'] for rep in reps]),
            dual_par_s=_mean([rep['attackers'][1]['dual_s'] for rep in reps]),
            solo_anc_c=_mean([rep['attackers'][0]['solo_c'] for rep in reps]),
            solo_par_c=_mean([rep['attackers'][1]['solo_c'] for rep in reps]),
            dual_anc_c=_mean([rep['attackers'][0]['dual_c'] for rep in reps]),
            dual_par_c=_mean([rep['attackers'][1]['dual_c'] for rep in reps]),
            d_ab=d_ab,
            cos_source=1.0 - d_ab,
            d_anc_tgt=a_self,
            d_par_tgt=b_self,
            a_cross=a_cross,
            b_cross=b_cross,
            target_target_distance=target_target_distance,
            cross_alignment_gap=cross_alignment_gap,
            cos_attack=cross_alignment_gap,
            cos_attack_dir=attack_direction_cos(first_a['source'], target_a, first_b['source'], target_b),
            motif_label=_c2_mini_motif(condition),
            source_stratum=source_stratum,
            alignment_type=alignment_type,
            condition=condition,
            same_source_clone=condition.endswith('__clone'),
        ))
    return payload


def _pair_key_from_pairing_id(pairing_id):
    if '_rep' in pairing_id:
        return pairing_id.rsplit('_rep', 1)[0]
    return pairing_id


def _c6_pair_metadata():
    if not C6_EXPERIMENT_PATH.exists():
        return {}

    experiment = json.loads(C6_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    metadata_by_key = {}
    for job in experiment.get('dual_jobs', []):
        pair_key = job.get('pair_key') or _pair_key_from_pairing_id(job.get('pairing_id', ''))
        if not pair_key:
            continue
        metadata_by_key[pair_key] = dict(
            pair_key=pair_key,
            pair_number=job.get('pair_number'),
            pair_bin=job.get('pair_bin') or job.get('distance_bucket'),
            target_pair_left_index=job.get('target_pair_left_index'),
            target_pair_right_index=job.get('target_pair_right_index'),
            shared_target_class_name=job.get('shared_target_class_name'),
            shared_poison_class_name=job.get('shared_poison_class_name'),
            feature_cosine_distance=job.get('feature_cosine_distance'),
            feature_cosine_similarity=job.get('feature_cosine_similarity'),
            target_target_distance=job.get('target_target_distance'),
            gradient_cosine=job.get('gradient_cosine'),
        )
    return metadata_by_key


def _c6_rows():
    dual_dir = C6_ROOT / 'dual'
    if not dual_dir.exists():
        return []

    solo_means = _solo_means_by_attacker(C6_ROOT)
    metadata_by_key = _c6_pair_metadata()
    by_condition = {}

    for path in sorted(dual_dir.glob('*.csv')):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        ordered_attacker_ids = list(dict.fromkeys(row['attacker_id'] for row in rows))
        if len(ordered_attacker_ids) != 2:
            raise ValueError(f'Expected two attackers in {path}.')

        pairing_id = rows[0]['pairing_id'] or path.stem.removeprefix('dual_')
        pair_key = _pair_key_from_pairing_id(pairing_id)
        pair_meta = dict(metadata_by_key.get(pair_key, {}))
        per_attacker = []
        for attacker_id in ordered_attacker_ids:
            if attacker_id not in solo_means:
                raise KeyError(f'Missing solo output for {attacker_id}.')
            attacker_rows = [row for row in rows if row['attacker_id'] == attacker_id]
            per_attacker.append(dict(
                source=attacker_rows[0]['target_true_class_name'],
                poison=attacker_rows[0]['target_adv_class_name'],
                source_target_distance=_float_or(attacker_rows[0]['source_target_distance'], 0.0),
                solo_s=solo_means[attacker_id]['success'],
                solo_c=solo_means[attacker_id]['confidence'],
                dual_s=_mean([row['success'] for row in attacker_rows]),
                dual_c=_mean([row['adv_confidence'] for row in attacker_rows]),
            ))

        first = rows[0]
        pair_meta.update(
            condition=first['condition'] or pair_key,
            distance_bucket=first['distance_bucket'] or pair_meta.get('pair_bin') or '',
            pair_number=first.get('pair_number') or pair_meta.get('pair_number'),
            selection_method=first.get('selection_method') or pair_meta.get('selection_method'),
            target_target_distance=_float_or(
                first.get('target_target_distance'),
                _float_or(pair_meta.get('target_target_distance'), 0.0),
            ),
            feature_cosine_distance=_float_or(
                first.get('feature_cosine_distance'),
                _float_or(pair_meta.get('feature_cosine_distance'), 0.0),
            ),
            feature_cosine_similarity=_float_or(
                first.get('feature_cosine_similarity'),
                _float_or(pair_meta.get('feature_cosine_similarity'), 0.0),
            ),
            gradient_cosine=_float_or(
                first.get('gradient_cosine'),
                _float_or(pair_meta.get('gradient_cosine'), 0.0),
            ),
        )
        by_condition.setdefault(pair_meta['condition'], []).append(dict(meta=pair_meta, attackers=per_attacker))

    payload = []
    for index, (condition, reps) in enumerate(sorted(by_condition.items())):
        first_meta = reps[0]['meta']
        first_a, first_b = reps[0]['attackers']
        bucket = first_meta.get('distance_bucket') or first_meta.get('pair_bin') or ''
        motif_label = f'c6_{bucket}' if bucket else 'c6_target_pair'
        source_label = first_a['source']
        poison_label = first_a['poison']

        a_success = [rep['attackers'][0]['dual_s'] - rep['attackers'][0]['solo_s'] for rep in reps]
        b_success = [rep['attackers'][1]['dual_s'] - rep['attackers'][1]['solo_s'] for rep in reps]
        a_confidence = [rep['attackers'][0]['dual_c'] - rep['attackers'][0]['solo_c'] for rep in reps]
        b_confidence = [rep['attackers'][1]['dual_c'] - rep['attackers'][1]['solo_c'] for rep in reps]
        source_target_distance = _mean([rep['attackers'][0]['source_target_distance'] for rep in reps])
        target_target_distances = [_float_or(rep['meta'].get('target_target_distance'), 0.0) for rep in reps]
        target_target_distance = _mean(target_target_distances)
        gradient_cosines = [_float_or(rep['meta'].get('gradient_cosine'), 0.0) for rep in reps]
        pair_keys = [rep['meta'].get('pair_key') for rep in reps if rep['meta'].get('pair_key')]
        pair_numbers = [rep['meta'].get('pair_number') for rep in reps if rep['meta'].get('pair_number') not in (None, '')]

        payload.append(dict(
            row_id=f'C6-{index}',
            dataset='C6',
            anchor=source_label,
            partner=source_label,
            target_class=poison_label,
            target_classes=[poison_label],
            target_a_class_name=poison_label,
            target_b_class_name=poison_label,
            n=len(reps),
            I_a_s=_mean(a_success),
            I_b_s=_mean(b_success),
            I_a_s_sem=_sem_values(a_success),
            I_b_s_sem=_sem_values(b_success),
            I_a_c=_mean(a_confidence),
            I_b_c=_mean(b_confidence),
            I_a_c_sem=_sem_values(a_confidence),
            I_b_c_sem=_sem_values(b_confidence),
            solo_anc_s=_mean([rep['attackers'][0]['solo_s'] for rep in reps]),
            solo_par_s=_mean([rep['attackers'][1]['solo_s'] for rep in reps]),
            dual_anc_s=_mean([rep['attackers'][0]['dual_s'] for rep in reps]),
            dual_par_s=_mean([rep['attackers'][1]['dual_s'] for rep in reps]),
            solo_anc_c=_mean([rep['attackers'][0]['solo_c'] for rep in reps]),
            solo_par_c=_mean([rep['attackers'][1]['solo_c'] for rep in reps]),
            dual_anc_c=_mean([rep['attackers'][0]['dual_c'] for rep in reps]),
            dual_par_c=_mean([rep['attackers'][1]['dual_c'] for rep in reps]),
            d_ab=target_target_distance,
            cos_source=1.0 - target_target_distance,
            d_anc_tgt=source_target_distance,
            d_par_tgt=source_target_distance,
            a_cross=None,
            b_cross=None,
            target_target_distance=target_target_distance,
            cross_alignment_gap=target_target_distance,
            gradient_cosine=_mean(gradient_cosines),
            cos_attack=target_target_distance,
            cos_attack_dir=target_target_distance,
            motif_label=motif_label,
            source_stratum=bucket,
            alignment_type='target_pair_similarity',
            condition=condition,
            pair_key=condition,
            pair_keys=pair_keys,
            pair_number=None,
            pair_numbers=pair_numbers,
            target_pair_left_index=None,
            target_pair_right_index=None,
        ))
    return payload


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    obj = torch.load(DIST_PATH, map_location='cpu', weights_only=False)
    centroids = obj['class_centroids'].numpy()
    centroid_idx = {name: i for i, name in enumerate(obj['class_names'])}

    def _unit(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _attack_direction_cos(source_a, target_a, source_b, target_b):
        va = _unit(centroids[centroid_idx[target_a]] - centroids[centroid_idx[source_a]])
        vb = _unit(centroids[centroid_idx[target_b]] - centroids[centroid_idx[source_b]])
        return float(np.dot(va, vb))

    c1 = build_cell_frame()
    c1_payload = []
    for index, record in enumerate(c1.to_dict(orient='records')):
        record.update(
            row_id=f'C1-{index}',
            cos_source=1.0 - record['d_ab'],
            target_class='airplane',
            target_classes=['airplane'],
            target_a_class_name='airplane',
            target_b_class_name='airplane',
        )
        c1_payload.append(record)

    for record in _same_source_clone_rows():
        record['row_id'] = f'C1-same-{record["anchor"]}'
        c1_payload.append(record)
    c1_payload.extend(_fork_far_apart_rows(_attack_direction_cos))

    def _sem(series):
        series = series.dropna()
        return float(series.std(ddof=1) / (len(series) ** 0.5)) if len(series) > 1 else 0.0

    c2 = compute_c2_interaction_frame(C2_EXPERIMENT_PATH)
    c2_group_cols = [
        'source_pair_label',
        'source_stratum',
        'motif_label',
        'alignment_type',
        'attacker_a_class_name',
        'attacker_b_class_name',
        'target_a_class_name',
        'target_b_class_name',
        'source_source_distance',
        'target_target_distance',
        'a_self',
        'b_self',
        'a_cross',
        'b_cross',
        'cross_alignment_gap',
    ]
    c2 = (
        c2.groupby(c2_group_cols, observed=True)
        .agg(
            n=('i_a', 'size'),
            i_a=('i_a', 'mean'),
            i_b=('i_b', 'mean'),
            i_a_sem=('i_a', _sem),
            i_b_sem=('i_b', _sem),
            i_a_s=('i_a_s', 'mean'),
            i_b_s=('i_b_s', 'mean'),
            i_a_s_sem=('i_a_s', _sem),
            i_b_s_sem=('i_b_s', _sem),
        )
        .reset_index()
    )
    c2_payload = []
    for index, record in enumerate(c2.to_dict(orient='records')):
        target_a = record['target_a_class_name']
        target_b = record['target_b_class_name']
        target_classes = list(dict.fromkeys([target_a, target_b]))
        target_label = target_a if target_a == target_b else f'{target_a} / {target_b}'
        c2_payload.append(dict(
            row_id=f'C2-{index}',
            dataset='C2',
            anchor=record['attacker_a_class_name'],
            partner=record['attacker_b_class_name'],
            target_class=target_label,
            target_classes=target_classes,
            target_a_class_name=target_a,
            target_b_class_name=target_b,
            n=record['n'],
            I_a_s=record['i_a_s'] / 100.0,
            I_b_s=record['i_b_s'] / 100.0,
            I_a_s_sem=record['i_a_s_sem'] / 100.0,
            I_b_s_sem=record['i_b_s_sem'] / 100.0,
            I_a_c=record['i_a'] / 100.0,
            I_b_c=record['i_b'] / 100.0,
            I_a_c_sem=record['i_a_sem'] / 100.0,
            I_b_c_sem=record['i_b_sem'] / 100.0,
            solo_anc_s=None,
            solo_par_s=None,
            dual_anc_s=None,
            dual_par_s=None,
            solo_anc_c=None,
            solo_par_c=None,
            dual_anc_c=None,
            dual_par_c=None,
            d_ab=record['source_source_distance'],
            cos_source=1.0 - record['source_source_distance'],
            d_anc_tgt=record['a_self'],
            d_par_tgt=record['b_self'],
            a_cross=record['a_cross'],
            b_cross=record['b_cross'],
            cross_alignment_gap=record['cross_alignment_gap'],
            cos_attack=record['cross_alignment_gap'],
            cos_attack_dir=_attack_direction_cos(
                record['attacker_a_class_name'],
                target_a,
                record['attacker_b_class_name'],
                target_b,
            ),
            motif_label=str(record['motif_label']),
            source_stratum=str(record['source_stratum']),
            alignment_type=record['alignment_type'],
            target_target_distance=record['target_target_distance'],
        ))

    dog_truck_override = _dog_cat_vs_truck_auto_override()
    for row in c2_payload:
        if (
            row['anchor'] == 'dog'
            and row['partner'] == 'truck'
            and row['target_a_class_name'] == 'cat'
            and row['target_b_class_name'] == 'automobile'
        ):
            row.update(dog_truck_override)
            break
    else:
        raise ValueError('Could not find C2 dog/truck cat/automobile row to update.')

    c2_mini_payload = _c2_mini_rows(_attack_direction_cos)
    c6_payload = _c6_rows()

    payload = {
        'C1': {
            'label': 'C1',
            'alignment_label': 'alignment',
            'rows': c1_payload,
        },
        'C2': {
            'label': 'C2',
            'alignment_label': 'cross-alignment gap',
            'rows': c2_payload,
        },
        'C2mini': {
            'label': 'C2 mini',
            'alignment_label': 'cross-alignment gap',
            'rows': c2_mini_payload,
        },
        'C6': {
            'label': 'C6',
            'alignment_label': 'target-pair distance',
            'rows': c6_payload,
        },
    }
    html = HTML.replace('__DATASETS_JSON__', json.dumps(payload))
    OUT.write_text(html, encoding='utf-8')
    print(
        f'wrote {OUT}  (C1={len(c1_payload)} cells, C2={len(c2_payload)} cells, '
        f'C2 mini={len(c2_mini_payload)} cells, C6={len(c6_payload)} cells)'
    )


if __name__ == '__main__':
    main()
