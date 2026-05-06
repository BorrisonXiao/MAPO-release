#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MAPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MAPO_ROOT="${MAPO_ROOT:-${DEFAULT_MAPO_ROOT}}"
export ROOT_DIR="${ROOT_DIR:-$(cd -- "${MAPO_ROOT}/.." && pwd)}"

PARSE_OPTIONS_SH="${MAPO_ROOT}/src/utils/parse_options.sh"
if [[ -f "${PARSE_OPTIONS_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${PARSE_OPTIONS_SH}"
fi

MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/.cache/downloads/model}"
MODEL_PATH="${model_path:-${MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Instruct-2507}}"

NODE_RANK="${node_rank:-${NODE_RANK:-0}}"
GPUS="${gpus:-${GPUS:-4}}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    # Default to the UPPER half of GPUs (4,5,6,7) to avoid colliding
    # with the rollout server that typically uses 0,1,2,3.
    FIRST_GPU=$((8 - GPUS))
    CUDA_VISIBLE_DEVICES="$(seq -s, "${FIRST_GPU}" "$((FIRST_GPU + GPUS - 1))")"
fi

PORT="${checker_port:-${port:-${CHECKER_PORT:-9000}}}"
TP="${tp:-${TP:-4}}"
GPU_MEMORY_UTILIZATION="${gpu_memory_utilization:-${GPU_MEMORY_UTILIZATION:-0.85}}"

CHECKER_PID=""

kill_tree() {
    local pid=$1 sig=${2:-TERM}
    for child in $(pgrep -P "${pid}" 2>/dev/null); do
        kill_tree "${child}" "${sig}"
    done
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    echo "Cleaning up..."
    if [[ -n "${CHECKER_PID}" ]]; then
        echo "Stopping checker process (PID: ${CHECKER_PID}) and all child processes..."
        kill_tree "${CHECKER_PID}" TERM
        sleep 3
        kill_tree "${CHECKER_PID}" 9
        echo "Checker process stopped."
    fi
}
trap cleanup EXIT INT TERM

LOG_DIR="${log_dir:-${LOG_DIR:-${MAPO_ROOT}/src/mapo/logs}}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
if [[ ! -w "$LOG_DIR" ]]; then
    echo "LOG_DIR is not writable: $LOG_DIR. Falling back to /tmp/mapo_logs." >&2
    LOG_DIR="/tmp/mapo_logs"
    mkdir -p "$LOG_DIR"
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting consistency-checker vLLM server on port ${PORT}..."
echo "Model: $MODEL_PATH"
echo "Using $GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "TP: $TP"

CHECKER_LOG="${LOG_DIR}/checker_${TIMESTAMP}.log"
echo "Saving checker log to: $CHECKER_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code 2>&1 | tee "$CHECKER_LOG" &

CHECKER_PID=$!
wait "$CHECKER_PID"
CHECKER_EXIT_CODE=$?
echo "Checker finished with exit code ${CHECKER_EXIT_CODE}."
exit ${CHECKER_EXIT_CODE}
