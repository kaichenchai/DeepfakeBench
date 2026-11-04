# DUG HPC Reference

Official reference extracted from DUG HPC documentation. See the [full docs](https://docs.hpc-portal.dug.com/insight-hpc/hpc/dug_hpc_user_manual/running_jobs.htm).

## Quick Commands

| Command | Purpose |
|---------|---------|
| `rj script.job` | Submit a job script to the queue |
| `squeue` | List your queued/running jobs |
| `squeue -u $USER` | List jobs for your user |
| `scancel JOBID` | Cancel a job (any state) |
| `scontrol update jobid=JOBID priority=N` | Change a job's priority |
| `jless JOBID` | View live log output (STDOUT+STDERR) for a job |
| `jcd JOBID` | Show working directory for a job |
| `srun --jobid JOBID --overlap --pty bash` | Get interactive shell on a running job's node |
| `scontrol release JOBID` | Rerun a job in SE (Special Exit) state |

## RJ Options Table

Options are specified in the script as `#rj <key>=<value>`.

| Option | Description | Default |
|--------|-------------|---------|
| `name` | Job name (defaults to script filename) | Script name |
| `queue` | Partition/queue to submit to (uses primary Unix group) | `*` |
| `priority` | Job priority, 1–1000 | — |
| `nodes` | Number of nodes for multi-node (e.g. MPI) jobs | — |
| `taskspernode` | Processes per node for multi-node jobs | — |
| `cpus` | CPUs requested (`*` = all CPUs on node) | — |
| `gpus` | GPU type and count (e.g. `a100:4`) | — |
| `mem` | Minimum memory per node (K, M, G suffixes) | — |
| `runtime` | Estimated runtime in hours (soft limit) | 24h (partition default) |
| `logdir` | Directory for job log output | `logs/` |
| `dep` | Job IDs to depend on (waits for them to complete) | — |
| `hold` | Set `1` to submit in held state (won't schedule until released) | — |
| `schema` | Schema file for array jobs | — |
| `array` | Subset of tasks from schema to run | All tasks |
| `export` | Comma-separated env vars to pass from submit shell | — |
| `features` | Node type filter (e.g. `a100`, `v100`, `mi50`, `zen`, `genoa`) | — |
| `io` | Set `1` for I/O-bound jobs (limits concurrent tasks) | — |
| `localdisk` | Require local disk: `1` (any) or size in GB | — |

## Job States

Jobs pass through states during execution. Typical flow: `PD` → `R` → `CD` (success) or `SE`/`F` (failure).

| Code | Name | Meaning |
|------|------|---------|
| `PD` | Pending | Awaiting resource allocation |
| `R` | Running | Currently executing with an allocation |
| `CD` | Completed | Terminated all processes with exit code 0 |
| `CG` | Completing | In the process of completing |
| `CF` | Configuring | Resources allocated, booting/readying |
| `SE` | Special Exit | Requeued in special state (job errored) |
| `F` | Failed | Non-zero exit code or other failure |
| `TO` | Timeout | Hit its time limit |
| `OOM` | Out of Memory | Exceeded memory allocation |
| `CA` | Cancelled | Explicitly cancelled by user or admin |
| `DL` | Deadline | Terminated on deadline |
| `NF` | Node Fail | Allocated node(s) failed |
| `PR` | Pre-empted | Terminated due to pre-emption |
| `RD` | RESV_DEL_HOLD | Job is held |
| `RH` | REQUEUE_HOLD | Held job being requeued |
| `RQ` | Requeued | Completing job being requeued |
| `RS` | Resizing | About to change size |
| `SI` | Signalling | Being signalled |
| `SO` | STAGE_OUT | Staging out files |
| `ST` | Stopped | SIGSTOP, CPUs retained |
| `S` | Suspended | Execution suspended, CPUs released |

## Node Types (features=)

| Keyword | Hardware |
|---------|----------|
| `a100` | Intel Ice Lake + Nvidia A100 GPUs (2×16-core CPU, 4 GPUs, 1TB RAM) |
| `v100` | Intel Cascade Lake + Nvidia V100 GPUs (2×20-core CPU, 4 GPUs, 384GB RAM) |
| `mi50` | AMD 2nd Gen Epyc + AMD mi50 GPUs (1×24-core CPU, 4 GPUs, 128–256GB RAM) |
| `zen` | AMD 2nd Gen Epyc (2×24-core CPU, 0.5TB RAM) |
| `genoa` | AMD 4th Gen Epyc (2×96-core CPU, 1.5TB RAM) |
| `icx` | Intel Ice Lake (2×16-core CPU, 2TB RAM) |
| `clx` | Intel Cascade Lake (2×24-core CPU, 384GB RAM) |
| `knl` | Intel Knights Landing (1×64-68-core CPU, 128–192GB RAM) |

## Job Output & Logging

- Default log directory: `logs/` (overridable with `logdir=`)
- Each job log contains: PROLOG (start timestamp, TMPDIR, node info) → job STDOUT/STDERR → EPILOG (runtime, TMPDIR cleanup)
- Log naming: `<job_name>.o<JOBID>` for single jobs, `<job_name>.o<JOBID>.<ARRAY_TASKID>` for array jobs
- View logs with `jless JOBID`

## TMPDIR

- Set automatically by Slurm — points to local disk or temp directory on file server
- Automatically deleted after job exits
- Move any files you want to keep to persistent storage at the end of your script
