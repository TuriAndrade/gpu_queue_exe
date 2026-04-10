# gpu_queue_exe

A lightweight Bash utility to execute a list of GPU jobs while respecting available GPU memory. It reads commands from a text file, monitors the selected GPUs with `nvidia-smi`, launches jobs when enough memory is available, and automatically requeues failed jobs with a higher memory requirement.

## What it does

- Schedules jobs from a plain text file, one command per line
- Works across one or more selected GPUs
- Launches jobs only when the required free memory is available
- Tracks running jobs and stores stdout/stderr logs
- Requeues failed jobs with a larger minimum memory threshold
- Uses `CUDA_VISIBLE_DEVICES` to bind each launched job to a specific GPU

## Requirements

- Bash
- `nvidia-smi`
- One or more NVIDIA GPUs
- A text file containing the commands you want to run

## Usage

```bash
gpu_queue_exe --devices 0,1 --min-memory 12000 --runs-file file1.txt --runs-file file2.txt [options]
```

## Required arguments

- `--devices`  
  Comma-separated physical GPU IDs. Examples: `0,1` or `cuda:0,cuda:1`

- `--min-memory`  
  Initial minimum free memory, in MiB, required to launch each job

- `--runs-file`  
  Path to a file containing one command per line

## Optional arguments

- `--offset`  
  Safety offset in MiB used when computing the maximum allowed launch memory  
  Default: `1024`

- `--retry-factor`  
  Multiplicative factor applied to a job's minimum memory requirement after failure  
  Must be greater than `1`  
  Default: `2.0`

- `--poll-seconds`  
  Time in seconds between GPU state checks  
  Default: `20`

- `--stabilization-delay`  
  Time in seconds a running job must keep the same observed GPU memory before its GPU is considered stable for another launch  
  Default: `20`

- `--max-jobs`  
  Maximum number of jobs allowed simultaneously on the machine  
  Use `0` for no machine-wide cap  
  Default: `0`

- `--logs-dir`  
  Directory where stdout and stderr logs are saved  
  Default: `./logs`

- `--workdir`  
  Working directory used before launching jobs  
  Default: current directory

## Example

### `runs.txt`

```text
python train.py --config config_a.json
python train.py --config config_b.json
python train.py --config config_c.json
```

## Command

```bash
gpu_queue_exe --devices 0,1 --min-memory 12000 --runs-file runs.txt [options]
```

## How it works

1. Reads all non-empty, non-comment lines from the runs file
2. Queries the selected GPUs for total and used memory
3. Computes free memory on each GPU
4. Launches the next job whose memory requirement fits on one of the available GPUs
5. Exports `CUDA_VISIBLE_DEVICES=<gpu_id>` for the launched process
6. Writes stdout and stderr to separate log files
7. If a job fails, it is requeued with a higher minimum memory threshold
8. Stops retrying once the job reaches the maximum allowed launch memory

## Log files

Each launched job generates two files:

- `job_<job_id>_attempt_<attempt>_gpu<gpu_id>.out`
- `job_<job_id>_attempt_<attempt>_gpu<gpu_id>.err`

These files are stored in the directory passed with `--logs-dir`.

## Notes

- Commands are executed exactly as written in the runs file
- Empty lines and lines starting with `#` are ignored
- The script does not modify your commands; it only wraps execution with `CUDA_VISIBLE_DEVICES`
- Requeued jobs are appended to the end of the queue
- A job that fails at the maximum allowed launch memory is not requeued again

## Minimal example for sourcing

```bash
source ./gpu_queue_exe.sh

gpu_queue_exe \
  --devices 0,1 \
  --min-memory 12000 \
  --runs-file ./runs.txt
```

## Direct execution

```bash
chmod +x gpu_queue_exe.sh
./gpu_queue_exe.sh --devices 0,1 --min-memory 12000 --runs-file ./runs.txt
```
