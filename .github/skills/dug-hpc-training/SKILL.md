---
name: dug-hpc-training
description: 'Run deepfake detection experiments on the DUG HPC cluster. Use when: run experiment, train model, submit job to DUG, create Slurm job, run on supercomputer.'
argument-hint: '[detector-config] [datasets...]'
user-invocable: true
disable-model-invocation: false
---

# DUG HPC Training

Run a DeepfakeBench training experiment on the DUG supercomputer. The repo is already set up on DUG — only worry about the detector config and the job file.

For full DUG HPC documentation (all `#rj` options, complete job states, node types), see [DUG HPC Reference](./references/dug-hpc-reference.md).

## Workflow

> **Important:** All commands on DUG must be run from the **repo root directory** (`DeepfakeBench/`) with the `uv` virtual environment activated (`source .venv/bin/activate`).

### Step 1: Prepare the Detector Config

Find an existing config in `training/config/detector/` that's similar to what you want, and copy it with a descriptive name (usually matching the config name, e.g. `effort_ce_hsic_orthogonal.yaml`).

**Critical:** Set both batch sizes to 64:

```yaml
train_batchSize: 64
test_batchSize: 64
```

### Step 2: Edit `train_ddp.job`

Update the job file to point to your config and desired datasets:

- `#rj name=` — set to a descriptive name matching the config
- `--detector_path` — path to the config from Step 1
- `--train_dataset` — training dataset(s)
- `--test_dataset` — test dataset(s)

Available datasets: `FaceForensics++`, `FF-F2F`, `FF-DF`, `FF-FS`, `FF-NT`, `FaceShifter`, `DeepFakeDetection`, `Celeb-DF-v1`, `Celeb-DF-v2`, `DFDCP`, `DFDC`, `DeeperForensics-1.0`, `UADFV`.

The dataset to train on is FaceForensics++ (FF++) by default, but you may want to test that something is working on a smaller dataset first (e.g. `UADFV` or `Celeb-DF-v1`). Make sure that the dataset actually exists on DUG before submitting the job. Check with `ls .datasets/rgb/` to see available datasets.

### Step 3: Commit to GitHub

```bash
git add -A && git commit -m "experiment: <description>" && git push
```

### Step 4: Submit on DUG

SSH into DUG, pull, activate the environment, and submit. **You must run `rj` from the repo root** with the venv active:

```bash
cd /path/to/DeepfakeBench   # must be in repo root
git pull
source .venv/bin/activate   # activate uv virtual environment first
rj train_ddp.job
```

### Step 5: Verify It's Running

```bash
squeue
```

The job state must be **`R`** (RUNNING). If it's `PD` (pending), or the job doesn't appear — stop and tell the user immediately.

**DUG job states reference:**

| Code | Meaning |
|------|---------|
| `R` | Running — job is allocated and executing |
| `PD` | Pending — waiting for resources |
| `SE` | Special Exit — job failed/errored; investigate with `jless JOBID` |
| `F` | Failed — terminated with non-zero exit code |
| `TO` | Timeout — hit its time limit |
| `OOM` | Out of memory |
| `CA` | Cancelled by user or admin |
| `CD` | Completed successfully |

**Useful commands:**

| Command | Purpose |
|---------|---------|
| `squeue` | Show your queued/running jobs |
| `jless JOBID` | View live log output for a job |
| `scancel JOBID` | Cancel a job |
| `scontrol update jobid=JOBID priority=N` | Change a job's priority |

## Job File Template (`train_ddp.job`)

```bash
#!/bin/bash
#rj name=kai_training
#rj gpus=a100:4
#rj nodes=1
#rj taskspernode=1
#rj cpus=*
#rj mem=200G
#rj runtime=1
#rj logdir=logs/slurm
#rj priority=550

# Bash Strict Mode
set -euo pipefail

# Activate virtual environment managed by uv (MUST be in repo root)
source .venv/bin/activate

# Redirect multiprocessing temp dir to local storage to avoid
# "Device or resource busy" errors on Lustre during worker cleanup
export TMPDIR=/tmp/slurm_job_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# Execute training using torchrun
# nproc_per_node=4 matches the 4 GPUs requested above
torchrun --nproc_per_node=4 \
    training/train.py \
    --detector_path ./training/config/detector/<config>.yaml \
    --train_dataset "<TrainSet>" \
    --test_dataset "<TestSet>" \
    --ddp

# Clean up local temp directory
rm -rf "$TMPDIR"

echo "Training job finished."
```

### `#rj` Directives

| Directive | Meaning |
|-----------|---------|
| `#rj name=` | Unique job name |
| `#rj gpus=a100:4` | 4 NVIDIA A100 GPUs |
| `#rj runtime=1` | Max 1 day |
| `#rj priority=550` | Higher = sooner (max 1000) |
