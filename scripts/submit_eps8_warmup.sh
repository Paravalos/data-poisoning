#!/bin/bash
mkdir -p artifacts/c1/c1_experiment_eps8/slurm
sbatch \
  --parsable \
  --account aip-yiweilu \
  --nodes 1 \
  --gres gpu:l40s:1 \
  --mem 20G \
  --cpus-per-task 4 \
  --time 0:30:00 \
  --job-name brew_src5_sel0_to0 \
  --output artifacts/c1/c1_experiment_eps8/slurm/brew_src5_sel0_to0-%j.out \
  --error artifacts/c1/c1_experiment_eps8/slurm/brew_src5_sel0_to0-%j.err \
  --wrap '/project/6113619/cparaval/data-poisoning/venv/bin/python /project/6113619/cparaval/data-poisoning/scripts/run_dual_attack_experiment.py --experiment artifacts/c1/c1_experiment_eps8.json --stage brew --job-id brew_src5_sel0_to0'
