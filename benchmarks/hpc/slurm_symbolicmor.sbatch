#!/bin/bash
#
# Template SLURM script for running SymbolicMOR benchmark campaigns.
# Adjust partition, module names, project path, and worker counts for the cluster.

#SBATCH --job-name=symbolicmor
#SBATCH --output=benchmark_results/symbolicmor_%j.out
#SBATCH --error=benchmark_results/symbolicmor_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# Example only. Replace with the cluster's Julia setup.
# module load julia/1.11

cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark_results

WORKERS="${WORKERS:-8}"
CASE="${CASE:-lorenz}"
TRAJECTORIES="${TRAJECTORIES:-512}"
TEND="${TEND:-10.0}"
DT="${DT:-0.02}"
REPEATS="${REPEATS:-3}"
OUT="benchmark_results/${CASE}_${WORKERS}w_${SLURM_JOB_ID}.csv"

julia --project=. -p "${WORKERS}" benchmarks/cluster_campaign.jl \
  --case "${CASE}" \
  --trajectories "${TRAJECTORIES}" \
  --tend "${TEND}" \
  --dt "${DT}" \
  --repeats "${REPEATS}" \
  --out "${OUT}"
