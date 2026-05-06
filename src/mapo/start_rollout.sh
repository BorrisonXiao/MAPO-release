#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MAPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MAPO_ROOT="${MAPO_ROOT:-${DEFAULT_MAPO_ROOT}}"
export ROOT_DIR="${ROOT_DIR:-$(cd -- "${MAPO_ROOT}/.." && pwd)}"

PARSE_OPTIONS_SH="${MAPO_ROOT}/src/utils/parse_options.sh"
if [[ ! -f "${PARSE_OPTIONS_SH}" ]]; then
    echo "parse_options.sh not found: ${PARSE_OPTIONS_SH}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${PARSE_OPTIONS_SH}"

MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/.cache/downloads/model}"
# MODEL_PATH="${model_path:-${MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Instruct}}"
MODEL_PATH="${model_path:-${MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Thinking}}"

NODE_RANK="${node_rank:-${NODE_RANK:-0}}"
GPUS_PER_NODE="${gpus_per_node:-${GPUS_PER_NODE:-${gpus:-${GPUS:-4}}}}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS_PER_NODE - 1))")"
fi

VLLM_PORT="${vllm_port:-${port:-${VLLM_SERVER_PORT:-8050}}}"
TP="${tp:-${TP:-4}}"
DP="${dp:-${DP:-1}}"
VLLM_GPU_MEMORY_UTILIZATION="${vllm_gpu_memory_utilization:-${VLLM_GPU_MEMORY_UTILIZATION:-0.75}}"

NCCL_DEBUG="${nccl_debug:-${NCCL_DEBUG:-INFO}}"
NCCL_DEBUG_SUBSYS="${nccl_debug_subsys:-${NCCL_DEBUG_SUBSYS:-ALL}}"
NCCL_SOCKET_IFNAME_VAL="${nccl_socket_ifname:-${NCCL_SOCKET_IFNAME:-eth1}}"
NCCL_IB_DISABLE_VAL="${nccl_ib_disable:-${NCCL_IB_DISABLE:-1}}"

ROLLOUT_PID=""

kill_tree() {
    local pid=$1 sig=${2:-TERM}
    for child in $(pgrep -P "${pid}" 2>/dev/null); do
        kill_tree "${child}" "${sig}"
    done
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    echo "Cleaning up..."
    if [[ -n "${ROLLOUT_PID}" ]]; then
        echo "Stopping rollout process (PID: ${ROLLOUT_PID}) and all child processes..."
        kill_tree "${ROLLOUT_PID}" TERM
        sleep 3
        kill_tree "${ROLLOUT_PID}" 9
        echo "Rollout process stopped."
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

export NCCL_DEBUG NCCL_DEBUG_SUBSYS
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_rollout_node${NODE_RANK}_${TIMESTAMP}.log"
if [[ -n "${NCCL_SOCKET_IFNAME_VAL}" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VAL}"
fi
if [[ -n "${NCCL_IB_DISABLE_VAL}" ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE_VAL}"
fi

echo "Starting vLLM rollout server on port ${VLLM_PORT}..."
echo "Model: $MODEL_PATH"
echo "Using $GPUS_PER_NODE GPUs: $CUDA_VISIBLE_DEVICES"
echo "TP: $TP, DP: $DP"

ROLLOUT_LOG="${LOG_DIR}/rollout_${TIMESTAMP}.log"
echo "Saving rollout log to: $ROLLOUT_LOG"
echo "Saving NCCL logs to: ${NCCL_DEBUG_FILE}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    swift rollout \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --vllm_tensor_parallel_size "$TP" \
    --vllm_data_parallel_size "$DP" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --port "$VLLM_PORT" 2>&1 | tee "$ROLLOUT_LOG" &

ROLLOUT_PID=$!
wait "$ROLLOUT_PID"
ROLLOUT_EXIT_CODE=$?
echo "Rollout finished with exit code ${ROLLOUT_EXIT_CODE}."
exit ${ROLLOUT_EXIT_CODE}
