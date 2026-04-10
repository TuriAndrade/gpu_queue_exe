#!/usr/bin/env bash

set -uo pipefail

gpu_queue_exe() {
  gqe__init_defaults
  gqe__parse_args "$@"
  gqe__validate_args
  gqe__run
}

gqe__init_defaults() {
  declare -ga GQE_RUNS_FILES=()
  GQE_DEVICES_ARG=""
  GQE_GLOBAL_MIN_MEMORY=""
  GQE_OFFSET_MB=1024
  GQE_RETRY_FACTOR=2.0
  GQE_POLL_SECONDS=20
  GQE_LOGS_DIR="./logs"
  GQE_WORKDIR=""
}

gqe__usage() {
  cat <<'USAGE'
Usage:
  gpu_queue_exe --devices 0,1 --min-memory 12000 --runs-file file1.txt --runs-file file2.txt [options]

Required:
  --devices              Comma-separated physical GPU ids, e.g. 0,1 or cuda:0,cuda:1
  --min-memory           Initial minimum free memory (MiB) required to launch each job
  --runs-file            File with one command per line; may be passed multiple times

Optional:
  --offset               Safety offset in MiB when capping retried jobs and updating min memory
                         default: 1024
  --retry-factor         Multiplicative factor applied to min memory after a failed job
                         must be > 1
                         default: 2.0
  --poll-seconds         Poll interval for GPU memory changes
                         default: 20
  --logs-dir             Directory for stdout/stderr logs
                         default: ./logs
  --workdir              Working directory before launching jobs
                         default: current directory

Examples:
  gpu_queue_exe \
    --devices 0,1 \
    --min-memory 12000 \
    --runs-file ./scripts/runs.txt \
    --logs-dir ./logs

  gpu_queue_exe \
    --devices cuda:0,cuda:1 \
    --min-memory 10000 \
    --runs-file ./runs.txt \
USAGE
}

gqe__die() {
  echo "[gpu_queue_exe] ERROR: $*" >&2
  return 1
}

gqe__parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --devices)
        GQE_DEVICES_ARG="${2:-}"
        shift 2
        ;;
      --min-memory)
        GQE_GLOBAL_MIN_MEMORY="${2:-}"
        shift 2
        ;;
      --runs-file)
        GQE_RUNS_FILES+=("${2:-}")
        shift 2
        ;;
      --offset)
        GQE_OFFSET_MB="${2:-}"
        shift 2
        ;;
      --retry-factor)
        GQE_RETRY_FACTOR="${2:-}"
        shift 2
        ;;
      --poll-seconds)
        GQE_POLL_SECONDS="${2:-}"
        shift 2
        ;;
      --logs-dir)
        GQE_LOGS_DIR="${2:-}"
        shift 2
        ;;
      --workdir)
        GQE_WORKDIR="${2:-}"
        shift 2
        ;;
      -h|--help)
        gqe__usage
        return 0
        ;;
      *)
        gqe__die "Unknown argument: $1" || return 1
        ;;
    esac
  done
}

gqe__validate_args() {
  [[ -n "$GQE_DEVICES_ARG" ]] || gqe__die "--devices is required" || return 1
  [[ -n "$GQE_GLOBAL_MIN_MEMORY" ]] || gqe__die "--min-memory is required" || return 1
  (( ${#GQE_RUNS_FILES[@]} > 0 )) || gqe__die "At least one --runs-file is required" || return 1

  [[ "$GQE_GLOBAL_MIN_MEMORY" =~ ^[0-9]+$ ]] || gqe__die "--min-memory must be an integer (MiB)" || return 1
  [[ "$GQE_OFFSET_MB" =~ ^[0-9]+$ ]] || gqe__die "--offset must be an integer (MiB)" || return 1
  [[ "$GQE_RETRY_FACTOR" =~ ^([0-9]+|[0-9]*\.[0-9]+)$ ]] || gqe__die "--retry-factor must be numeric" || return 1
  awk -v x="$GQE_RETRY_FACTOR" 'BEGIN { exit !(x > 1) }' || gqe__die "--retry-factor must be > 1" || return 1
  [[ "$GQE_POLL_SECONDS" =~ ^([0-9]+|[0-9]*\.[0-9]+)$ ]] || gqe__die "--poll-seconds must be numeric" || return 1

  local runs_file
  for runs_file in "${GQE_RUNS_FILES[@]}"; do
    [[ -f "$runs_file" ]] || gqe__die "Runs file not found: $runs_file" || return 1
  done
  command -v nvidia-smi >/dev/null 2>&1 || gqe__die "nvidia-smi not found" || return 1

  GQE_DEVICE_IDS=()
  local token
  IFS=',' read -r -a raw_devices <<< "$GQE_DEVICES_ARG"
  for token in "${raw_devices[@]}"; do
    token="${token//[[:space:]]/}"
    token="${token#cuda:}"
    [[ "$token" =~ ^[0-9]+$ ]] || gqe__die "Invalid device token: $token" || return 1
    GQE_DEVICE_IDS+=("$token")
  done

  (( ${#GQE_DEVICE_IDS[@]} > 0 )) || gqe__die "No valid devices parsed from --devices" || return 1
}

gqe__query_gpu_state() {
  local gpu_csv out idx total used
  local app_out pid curr_used old_min

  gpu_csv=$(IFS=,; echo "${GQE_DEVICE_IDS[*]}")

  out=$(nvidia-smi -i "$gpu_csv" \
    --query-gpu=index,memory.total,memory.used \
    --format=csv,noheader,nounits)

  GQE_GPU_SNAPSHOT="$out"
  unset GQE_GPU_TOTAL
  declare -gA GQE_GPU_TOTAL=()

  while IFS=',' read -r idx total used; do
    idx="${idx//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    [[ -n "$idx" ]] || continue

    GQE_GPU_TOTAL["$idx"]="$total"
  done <<< "$out"

  unset GQE_PID_USED
  declare -gA GQE_PID_USED=()

  app_out=$(nvidia-smi \
    --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)

  while IFS=',' read -r pid curr_used; do
    pid="${pid//[[:space:]]/}"
    curr_used="${curr_used//[[:space:]]/}"
    [[ -n "$pid" && -n "$curr_used" ]] || continue
    GQE_PID_USED["$pid"]="$curr_used"
  done <<< "$app_out"

  for pid in "${GQE_ACTIVE_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    curr_used="${GQE_PID_USED[$pid]:-}"
    [[ -n "$curr_used" ]] || continue
    old_min="${GQE_PID_TO_MIN[$pid]}"
    if (( curr_used > old_min )); then
      GQE_PID_TO_MIN["$pid"]="$curr_used"
    fi
  done
}

gqe__append_queue() {
  local job_id="$1"
  local cmd="$2"
  local min_mem="$3"
  local attempt="$4"

  GQE_QUEUE_IDS+=("$job_id")
  GQE_QUEUE_CMDS+=("$cmd")
  GQE_QUEUE_MINS+=("$min_mem")
  GQE_QUEUE_ATTEMPTS+=("$attempt")
}

gqe__load_queue() {
  unset GQE_QUEUE_IDS GQE_QUEUE_CMDS GQE_QUEUE_MINS GQE_QUEUE_ATTEMPTS
  declare -ga GQE_QUEUE_IDS=()
  declare -ga GQE_QUEUE_CMDS=()
  declare -ga GQE_QUEUE_MINS=()
  declare -ga GQE_QUEUE_ATTEMPTS=()

  local cmd job_counter=0 runs_file
  for runs_file in "${GQE_RUNS_FILES[@]}"; do
    while IFS= read -r cmd || [[ -n "$cmd" ]]; do
      [[ -z "${cmd//[[:space:]]/}" ]] && continue
      [[ "$cmd" =~ ^[[:space:]]*# ]] && continue
      job_counter=$((job_counter + 1))
      gqe__append_queue "$job_counter" "$cmd" "$GQE_GLOBAL_MIN_MEMORY" 0
    done < "$runs_file"
  done

  (( ${#GQE_QUEUE_CMDS[@]} > 0 )) || gqe__die "No runnable commands found in provided runs files" || return 1
}

gqe__compute_max_launch_mem() {
  local dev cap
  GQE_MAX_LAUNCH_MEM=0

  gqe__query_gpu_state || return 1

  for dev in "${GQE_DEVICE_IDS[@]}"; do
    [[ -n "${GQE_GPU_TOTAL[$dev]:-}" ]] || gqe__die "Could not query GPU $dev" || return 1
    cap=$(( GQE_GPU_TOTAL[$dev] - GQE_OFFSET_MB ))
    if (( cap > GQE_MAX_LAUNCH_MEM )); then
      GQE_MAX_LAUNCH_MEM=$cap
    fi
  done

  (( GQE_MAX_LAUNCH_MEM > 0 )) || gqe__die "All selected GPUs have total memory <= offset" || return 1
  (( GQE_GLOBAL_MIN_MEMORY <= GQE_MAX_LAUNCH_MEM )) || \
    gqe__die "--min-memory exceeds max allowed launch memory (${GQE_MAX_LAUNCH_MEM} MiB)" || return 1
}

gqe__build_launch_cmd() {
  local cmd="$1"
  local physical_dev="$2"

  printf '%s' "$cmd"
}

gqe__launch_job() {
  local dev="$1"
  local job_id="$2"
  local cmd="$3"
  local min_mem="$4"
  local attempt="$5"

  local out_log="${GQE_LOGS_DIR}/job_${job_id}_attempt_${attempt}_gpu${dev}.out"
  local err_log="${GQE_LOGS_DIR}/job_${job_id}_attempt_${attempt}_gpu${dev}.err"
  local full_cmd
  full_cmd=$(gqe__build_launch_cmd "$cmd" "$dev")

  echo "[gpu_queue_exe] Launching job=$job_id attempt=$attempt gpu=$dev min_mem=${min_mem}MiB"
  CUDA_VISIBLE_DEVICES="$dev" bash -c "exec $full_cmd" >"$out_log" 2>"$err_log" &
  local pid=$!

  GQE_ACTIVE_PIDS+=("$pid")
  GQE_PID_TO_DEV["$pid"]="$dev"
  GQE_PID_TO_CMD["$pid"]="$cmd"
  GQE_PID_TO_MIN["$pid"]="$min_mem"
  GQE_PID_TO_JOBID["$pid"]="$job_id"
  GQE_PID_TO_ATTEMPT["$pid"]="$attempt"
  GQE_PID_TO_OUT["$pid"]="$out_log"
  GQE_PID_TO_ERR["$pid"]="$err_log"
}

gqe__harvest_finished_jobs() {
  GQE_HARVESTED_ANY=0
  local -a still_running=()
  local pid pstate status dev cmd old_min job_id attempt out_log err_log new_min next_attempt

  for pid in "${GQE_ACTIVE_PIDS[@]}"; do
    pstate=$(ps -p "$pid" -o stat= 2>/dev/null || true)
    pstate="${pstate//[[:space:]]/}"

    if [[ -n "$pstate" && "$pstate" != Z* ]]; then
      still_running+=("$pid")
      continue
    fi

    GQE_HARVESTED_ANY=1

    if wait "$pid"; then
      status=0
    else
      status=$?
    fi

    dev="${GQE_PID_TO_DEV[$pid]}"
    cmd="${GQE_PID_TO_CMD[$pid]}"
    old_min="${GQE_PID_TO_MIN[$pid]}"
    job_id="${GQE_PID_TO_JOBID[$pid]}"
    attempt="${GQE_PID_TO_ATTEMPT[$pid]}"
    out_log="${GQE_PID_TO_OUT[$pid]}"
    err_log="${GQE_PID_TO_ERR[$pid]}"

    if (( status == 0 )); then
      echo "[gpu_queue_exe] Finished job=$job_id attempt=$attempt gpu=$dev status=0 out=$out_log err=$err_log"
    else
      if (( old_min >= GQE_MAX_LAUNCH_MEM )); then
        echo "[gpu_queue_exe] Job=$job_id failed with status=$status at max launch memory (${GQE_MAX_LAUNCH_MEM}MiB); not requeueing. out=$out_log err=$err_log"
      else
        new_min=$(awk -v old="$old_min" -v factor="$GQE_RETRY_FACTOR" 'BEGIN { printf "%d", int(old * factor + 0.999999) }')
        if (( new_min > GQE_MAX_LAUNCH_MEM )); then
          new_min=$GQE_MAX_LAUNCH_MEM
        fi

        next_attempt=$((attempt + 1))
        gqe__append_queue "$job_id" "$cmd" "$new_min" "$next_attempt"
        echo "[gpu_queue_exe] Requeued job=$job_id after status=$status old_min=${old_min}MiB new_min=${new_min}MiB out=$out_log err=$err_log"
      fi
    fi

    unset GQE_PID_TO_DEV["$pid"]
    unset GQE_PID_TO_CMD["$pid"]
    unset GQE_PID_TO_MIN["$pid"]
    unset GQE_PID_TO_JOBID["$pid"]
    unset GQE_PID_TO_ATTEMPT["$pid"]
    unset GQE_PID_TO_OUT["$pid"]
    unset GQE_PID_TO_ERR["$pid"]
  done

  GQE_ACTIVE_PIDS=("${still_running[@]}")
}

gqe__pick_best_device_for_min() {
  local min_mem="$1"
  local best_dev=""
  local dev free_mem total_a total_b used_a used_b

  for dev in "${GQE_DEVICE_IDS[@]}"; do
    free_mem="${GQE_RESERVED_FREE[$dev]}"
    (( free_mem >= min_mem )) || continue

    if [[ -z "$best_dev" ]]; then
      best_dev="$dev"
      continue
    fi

    total_a="${GQE_GPU_TOTAL[$dev]}"
    total_b="${GQE_GPU_TOTAL[$best_dev]}"
    used_a=$(( total_a - GQE_RESERVED_FREE[$dev] ))
    used_b=$(( total_b - GQE_RESERVED_FREE[$best_dev] ))

    if (( used_a * total_b < used_b * total_a )); then
      best_dev="$dev"
    elif (( used_a * total_b == used_b * total_a )); then
      if (( GQE_RESERVED_FREE[$dev] > GQE_RESERVED_FREE[$best_dev] )); then
        best_dev="$dev"
      fi
    fi
  done

  printf '%s' "$best_dev"
}

gqe__cleanup() {
  local pid
  for pid in "${GQE_ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}

gqe__run() {
  if [[ -n "$GQE_WORKDIR" ]]; then
    cd "$GQE_WORKDIR" || return 1
  fi
  mkdir -p "$GQE_LOGS_DIR" || return 1

  gqe__load_queue || return 1

  unset GQE_ACTIVE_PIDS
  declare -ga GQE_ACTIVE_PIDS=()

  gqe__compute_max_launch_mem || return 1

  unset GQE_PID_TO_DEV GQE_PID_TO_CMD GQE_PID_TO_MIN GQE_PID_TO_JOBID GQE_PID_TO_ATTEMPT GQE_PID_TO_OUT GQE_PID_TO_ERR
  declare -gA GQE_PID_TO_DEV=()
  declare -gA GQE_PID_TO_CMD=()
  declare -gA GQE_PID_TO_MIN=()
  declare -gA GQE_PID_TO_JOBID=()
  declare -gA GQE_PID_TO_ATTEMPT=()
  declare -gA GQE_PID_TO_OUT=()
  declare -gA GQE_PID_TO_ERR=()

  GQE_LAST_GPU_SNAPSHOT=""
  GQE_FIRST_DISPATCH=1

  trap gqe__cleanup INT TERM

  local dispatch_now launched_any chosen_idx chosen_dev idx min_mem job_id cmd attempt dev

  while (( ${#GQE_QUEUE_CMDS[@]} > 0 || ${#GQE_ACTIVE_PIDS[@]} > 0 )); do
    gqe__harvest_finished_jobs
    gqe__query_gpu_state

    dispatch_now=0
    if (( GQE_FIRST_DISPATCH )); then
      dispatch_now=1
    elif (( GQE_HARVESTED_ANY )); then
      dispatch_now=1
    elif [[ "$GQE_GPU_SNAPSHOT" != "$GQE_LAST_GPU_SNAPSHOT" ]]; then
      dispatch_now=1
    fi

    if (( dispatch_now )); then
      GQE_FIRST_DISPATCH=0
      GQE_LAST_GPU_SNAPSHOT="$GQE_GPU_SNAPSHOT"

      unset GQE_RESERVED_FREE
      declare -gA GQE_RESERVED_FREE=()

      for dev in "${GQE_DEVICE_IDS[@]}"; do
        GQE_RESERVED_FREE["$dev"]=$(( GQE_GPU_TOTAL[$dev] - GQE_OFFSET_MB ))
      done

      for pid in "${GQE_ACTIVE_PIDS[@]}"; do
        dev="${GQE_PID_TO_DEV[$pid]}"
        GQE_RESERVED_FREE["$dev"]=$(( GQE_RESERVED_FREE[$dev] - GQE_PID_TO_MIN[$pid] ))
      done

      launched_any=0

      while (( ${#GQE_QUEUE_CMDS[@]} > 0 )); do
        chosen_idx=-1
        chosen_dev=""

        for idx in "${!GQE_QUEUE_CMDS[@]}"; do
          min_mem="${GQE_QUEUE_MINS[$idx]}"
          chosen_dev=$(gqe__pick_best_device_for_min "$min_mem")
          if [[ -n "$chosen_dev" ]]; then
            chosen_idx="$idx"
            break
          fi
        done

        if (( chosen_idx < 0 )); then
          break
        fi

        job_id="${GQE_QUEUE_IDS[$chosen_idx]}"
        cmd="${GQE_QUEUE_CMDS[$chosen_idx]}"
        min_mem="${GQE_QUEUE_MINS[$chosen_idx]}"
        attempt="${GQE_QUEUE_ATTEMPTS[$chosen_idx]}"

        gqe__launch_job "$chosen_dev" "$job_id" "$cmd" "$min_mem" "$attempt"
        GQE_RESERVED_FREE["$chosen_dev"]=$(( GQE_RESERVED_FREE[$chosen_dev] - min_mem ))
        launched_any=1

        unset "GQE_QUEUE_IDS[$chosen_idx]"
        unset "GQE_QUEUE_CMDS[$chosen_idx]"
        unset "GQE_QUEUE_MINS[$chosen_idx]"
        unset "GQE_QUEUE_ATTEMPTS[$chosen_idx]"

        GQE_QUEUE_IDS=("${GQE_QUEUE_IDS[@]}")
        GQE_QUEUE_CMDS=("${GQE_QUEUE_CMDS[@]}")
        GQE_QUEUE_MINS=("${GQE_QUEUE_MINS[@]}")
        GQE_QUEUE_ATTEMPTS=("${GQE_QUEUE_ATTEMPTS[@]}")
      done

      if (( launched_any == 0 )) && (( ${#GQE_ACTIVE_PIDS[@]} > 0 || ${#GQE_QUEUE_CMDS[@]} > 0 )); then
        sleep "$GQE_POLL_SECONDS"
      fi
    else
      sleep "$GQE_POLL_SECONDS"
    fi
  done

  trap - INT TERM
  echo "[gpu_queue_exe] All runs finished."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  gpu_queue_exe "$@"
fi
