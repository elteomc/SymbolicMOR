# HPC Benchmark Notes

This folder contains portable templates for running the benchmark campaign on a
SLURM-style cluster, including MIT SuperCloud. The template is intentionally
conservative because module names, partitions, and filesystem layouts vary by cluster.

## Local Dry Run

From the repository root:

```bash
julia --project=. -p 1 benchmarks/cluster_campaign.jl --case lorenz --trajectories 16 --tend 1.0 --dt 0.05 --repeats 1 --out benchmark_results/local_p1.csv
julia --project=. -p 2 benchmarks/cluster_campaign.jl --case lorenz --trajectories 16 --tend 1.0 --dt 0.05 --repeats 1 --out benchmark_results/local_p2.csv
julia --project=. benchmarks/summarize_campaign.jl benchmark_results benchmark_results/local_summary
```

## SLURM Run

Edit `slurm_symbolicmor.sbatch` for the target system, especially:

- Julia module or executable path.
- Wall time.
- CPU count.
- Output directory.
- Case size.

Submit with:

```bash
sbatch benchmarks/hpc/slurm_symbolicmor.sbatch
```

Override common settings without editing the file:

```bash
WORKERS=16 CASE=lorenz TRAJECTORIES=2048 TEND=20.0 DT=0.02 REPEATS=3 sbatch benchmarks/hpc/slurm_symbolicmor.sbatch
```

## Result Schema

`benchmarks/cluster_campaign.jl` appends CSV rows with these columns:

- `case`
- `workers`
- `trajectories`
- `tend`
- `dt`
- `repeats`
- `serial_time`
- `parallel_time`
- `speedup`
- `efficiency`
- `julia_version`
- `os`
- `cpu`
- `git_sha`

Send the resulting CSV files back to a local machine and summarize them with:

```bash
julia --project=. benchmarks/summarize_campaign.jl benchmark_results benchmark_results/supercloud_summary
```
