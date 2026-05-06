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

# Pre-provision Megatron-LM with a clean linker env so ms-swift does not run
# git clone under a potentially incompatible LD_LIBRARY_PATH.
MEGATRON_LM_PATH="${megatron_lm_path:-${MEGATRON_LM_PATH:-${MAPO_ROOT}/libs/Megatron-LM}}"
MEGATRON_LM_BRANCH="${megatron_lm_branch:-${MEGATRON_LM_BRANCH:-core_r0.15.0}}"
MEGATRON_LM_REPO="${megatron_lm_repo:-${MEGATRON_LM_REPO:-https://github.com/NVIDIA/Megatron-LM.git}}"
if [[ ! -d "${MEGATRON_LM_PATH}/.git" ]]; then
    echo "Megatron-LM not found at ${MEGATRON_LM_PATH}. Cloning ${MEGATRON_LM_REPO} (branch ${MEGATRON_LM_BRANCH})..."
    mkdir -p "$(dirname -- "${MEGATRON_LM_PATH}")"
    if ! env -u LD_LIBRARY_PATH git -C "$(dirname -- "${MEGATRON_LM_PATH}")" clone \
        "${MEGATRON_LM_REPO}" "$(basename -- "${MEGATRON_LM_PATH}")" \
        --branch "${MEGATRON_LM_BRANCH}"; then
        echo "Failed to clone Megatron-LM with clean LD_LIBRARY_PATH." >&2
        exit 1
    fi
fi
export MEGATRON_LM_PATH
export MEGATRON_LM_REPO

DATA_ROOT="${DATA_ROOT:-${MAPO_ROOT}/data/}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/.cache/downloads/model}"
export NLTK_DATA="${nltk_data:-${NLTK_DATA:-${MODEL_ROOT}/nltk_data}}"

# MODEL_PATH="${model_path:-${MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Instruct}}"
MODEL_PATH="${model_path:-${MODEL_PATH:-${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Thinking}}"
DATASET_PATH="${dataset_path:-${DATASET_PATH:-${DATA_ROOT}/raw/vggsound/train_swift-for-omni-tags_with_caption_cot_cleaned.jsonl}}"
EXP_DIR="${exp_dir:-${EXP_DIR:-${MAPO_ROOT}/exp/mapo/${MODEL_PATH##*/}}}"

MASTER_ADDR="${master_node:-${master_addr:-${MASTER_ADDR:-${MASTER_NODE_IP:-}}}}"
MASTER_PORT="${master_port:-${MASTER_PORT:-29500}}"
NNODES="${nnodes:-${NNODES:-3}}"
NODE_RANK="${node_rank:-${NODE_RANK:-0}}"

GPUS_PER_NODE="${gpus_per_node:-${GPUS_PER_NODE:-${gpus:-${GPUS:-8}}}}"
NPROC_PER_NODE="${nproc_per_node:-${NPROC_PER_NODE:-${GPUS_PER_NODE}}}"
CUDA_VISIBLE_DEVICES="${cuda_visible_devices:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS_PER_NODE - 1))")"
fi

ROLLOUT_NODE="${rollout_node:-${vllm_server_host:-${ROLLOUT_NODE:-${ROLLOUT_NODE_IP:-${WORKER_NODE_1_IP:-${WORKER_NODE_2_IP:-}}}}}}"
VLLM_SERVER_PORT="${vllm_port:-${vllm_server_port:-${VLLM_SERVER_PORT:-${VLLM_PORT:-8050}}}}"
VLLM_SERVER_GROUP_PORT="${vllm_group_port:-${vllm_server_group_port:-${VLLM_SERVER_GROUP_PORT:-${VLLM_GROUP_PORT:-52011}}}}"

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "MASTER_ADDR is empty. Set MASTER_NODE_IP/MASTER_ADDR or pass --master-node." >&2
    exit 1
fi
if [[ -z "${ROLLOUT_NODE}" ]]; then
    echo "ROLLOUT_NODE is empty. Set ROLLOUT_NODE_IP/WORKER_NODE_1_IP or pass --rollout-node." >&2
    exit 1
fi

export MASTER_ADDR MASTER_PORT NNODES NODE_RANK
export GPUS_PER_NODE NPROC_PER_NODE CUDA_VISIBLE_DEVICES
export MODEL_PATH DATASET_PATH
export VLLM_SERVER_PORT VLLM_SERVER_GROUP_PORT

NCCL_DEBUG="${nccl_debug:-${NCCL_DEBUG:-INFO}}"
NCCL_DEBUG_SUBSYS="${nccl_debug_subsys:-${NCCL_DEBUG_SUBSYS:-ALL}}"
NCCL_SOCKET_IFNAME_VAL="${nccl_socket_ifname:-${NCCL_SOCKET_IFNAME:-eth1}}"
NCCL_IB_DISABLE_VAL="${nccl_ib_disable:-${NCCL_IB_DISABLE:-1}}"

# Logging setup
LOG_DIR="${log_dir:-${LOG_DIR:-${MAPO_ROOT}/src/mapo/logs}}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
if [[ ! -w "$LOG_DIR" ]]; then
    echo "LOG_DIR is not writable: $LOG_DIR. Falling back to /tmp/mapo_logs." >&2
    LOG_DIR="/tmp/mapo_logs"
    mkdir -p "$LOG_DIR"
fi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export NCCL_DEBUG NCCL_DEBUG_SUBSYS
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_train_node${NODE_RANK}_${TIMESTAMP}.log"
if [[ -n "${NCCL_SOCKET_IFNAME_VAL}" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VAL}"
fi
if [[ -n "${NCCL_IB_DISABLE_VAL}" ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE_VAL}"
fi

TRAIN_PID=""

kill_tree() {
    local pid=$1 sig=${2:-TERM}
    for child in $(pgrep -P "${pid}" 2>/dev/null); do
        kill_tree "${child}" "${sig}"
    done
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    echo "Cleaning up..."
    if [[ -n "${TRAIN_PID}" ]]; then
        echo "Stopping training process (PID: ${TRAIN_PID}) and all child processes..."
        kill_tree "${TRAIN_PID}" TERM
        sleep 3
        kill_tree "${TRAIN_PID}" 9
        echo "Training process stopped."
    fi
}
trap cleanup EXIT INT TERM

# SYSTEM_PROMPT=$(
#     cat <<'PROMPT'
# You are an audio QA expert. In <reasoning> tags, ALWAYS start with a brief, question-focused audio caption: describe sounds/speech/music/context relevant to answering the question. Then reason from that caption to your conclusion. End with <answer>exact option</answer>.

# Format:
# <reasoning>[question-focused caption] -> [reasoning] -> [evidence-based conclusion]</reasoning>
# <answer>[exact option text]</answer>
# PROMPT
# )
SYSTEM_PROMPT=""
PLUGIN_PATH="${plugin_path:-${PLUGIN_PATH:-${MAPO_ROOT}/src/rewards/audio_qa_rewards.py}}"
if [[ ! -f "${PLUGIN_PATH}" ]]; then
    echo "Reward plugin not found: ${PLUGIN_PATH}" >&2
    exit 1
fi

export MAPO_REWARD_VERBOSE="${mapo_reward_verbose:-${MAPO_REWARD_VERBOSE:-1}}"
export MAPO_REWARD_LOG_TO_FILE="${mapo_reward_log_to_file:-${MAPO_REWARD_LOG_TO_FILE:-1}}"
export PYTHONUNBUFFERED="${pythonunbuffered:-${PYTHONUNBUFFERED:-1}}"
export PYTHONDONTWRITEBYTECODE="${pythondontwritebytecode:-${PYTHONDONTWRITEBYTECODE:-1}}"

TP="${tp:-${TP:-4}}"
PP="${pp:-${PP:-2}}"
CP="${cp:-${CP:-1}}"
EP="${ep:-${EP:-2}}"

NUM_GENERATIONS="${num_generations:-${NUM_GENERATIONS:-8}}"
MICRO_BATCH_SIZE="${micro_batch_size:-${MICRO_BATCH_SIZE:-4}}"
GRAD_ACC="${grad_acc:-${GRAD_ACC:-2}}"

GLOBAL_BATCH_SIZE_VAL="${global_batch_size:-${GLOBAL_BATCH_SIZE:-}}"
if [[ -z "${GLOBAL_BATCH_SIZE_VAL}" ]]; then
    parallel_denom=$((TP * PP * CP))
    if (( parallel_denom <= 0 )); then
        echo "TP*PP*CP must be > 0, got ${parallel_denom}." >&2
        exit 1
    fi
    GLOBAL_BATCH_SIZE_VAL=$((NNODES * GPUS_PER_NODE / parallel_denom * MICRO_BATCH_SIZE * NUM_GENERATIONS * GRAD_ACC))
fi

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Rank: $NODE_RANK/$NNODES"
echo "Rollout Server: $ROLLOUT_NODE:$VLLM_SERVER_PORT"
echo "Rollout Group Port: $VLLM_SERVER_GROUP_PORT"
echo "Using $GPUS_PER_NODE GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "NLTK_DATA: $NLTK_DATA"
echo "Parallelism: TP=$TP, PP=$PP, CP=$CP, EP=$EP"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE_VAL"
echo "Reward Plugin: $PLUGIN_PATH"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export ENABLE_AUDIO_OUTPUT="${enable_audio_output:-${ENABLE_AUDIO_OUTPUT:-false}}"

TUNER_TYPE="${tuner_type:-${TUNER_TYPE:-lora}}"
LORA_RANK="${lora_rank:-${LORA_RANK:-128}}"
LORA_ALPHA="${lora_alpha:-${LORA_ALPHA:-512}}"
TARGET_MODULES="${target_modules:-${TARGET_MODULES:-all-linear}}"
TEMPLATE="${template:-${TEMPLATE:-qwen3_omni}}"
MAX_COMPLETION_LENGTH="${max_completion_length:-${MAX_COMPLETION_LENGTH:-1200}}"
STEPS_PER_GENERATION="${steps_per_generation:-${STEPS_PER_GENERATION:-4}}"
LR="${lr:-${LR:-1e-6}}"
LR_WARMUP_FRACTION="${lr_warmup_fraction:-${LR_WARMUP_FRACTION:-0.05}}"
BETA="${beta:-${BETA:-0.02}}"
ETA="${eta:-${ETA:-0.05}}"
MAPO_ADVANTAGE_FLOOR_EPS="${mapo_advantage_floor_eps:-${MAPO_ADVANTAGE_FLOOR_EPS:-0.02}}"
MAPO_TASK_FAIL_GATE_FLOOR="${mapo_task_fail_gate_floor:-${MAPO_TASK_FAIL_GATE_FLOOR:-0.1}}"
MAPO_ATTN_PREFACTOR_CLIP="${mapo_attn_prefactor_clip:-${MAPO_ATTN_PREFACTOR_CLIP:-3.0}}"
MAPO_MASK_TEMPERATURE="${mapo_mask_temperature:-${MAPO_MASK_TEMPERATURE:-1.0}}"
MAPO_MASK_CLIP="${mapo_mask_clip:-${MAPO_MASK_CLIP:-5.0}}"
MAPO_TEMPORAL_KAPPA="${mapo_temporal_kappa:-${MAPO_TEMPORAL_KAPPA:-1.0}}"
MAPO_ATTENTION_LAYERS="${mapo_attention_layers:-${MAPO_ATTENTION_LAYERS:-45,46,47}}"
MAPO_ATTENTION_HEAD_REDUCE="${mapo_attention_head_reduce:-${MAPO_ATTENTION_HEAD_REDUCE:-max}}"
MAPO_ATTENTION_LAYER_REDUCE="${mapo_attention_layer_reduce:-${MAPO_ATTENTION_LAYER_REDUCE:-mean}}"
MAPO_FAILURE_REWARD_NAME="${mapo_failure_reward_name:-${MAPO_FAILURE_REWARD_NAME:-MCQAAccuracy}}"
MAPO_FAILURE_THRESHOLD="${mapo_failure_threshold:-${MAPO_FAILURE_THRESHOLD:-0.0}}"
MAPO_POS_TAGS="${mapo_pos_tags:-${MAPO_POS_TAGS:-NOUN,VERB,ADJ,ADV,NUM,X}}"
MAPO_ATTENTION_ONLY="${mapo_attention_only:-${MAPO_ATTENTION_ONLY:-false}}"
MAPO_DEBUG_ATTN_GRAD_PROBE="${mapo_debug_attn_grad_probe:-${MAPO_DEBUG_ATTN_GRAD_PROBE:-false}}"
MAPO_DEBUG_ATTN_GRAD_PROBE_INTERVAL="${mapo_debug_attn_grad_probe_interval:-${MAPO_DEBUG_ATTN_GRAD_PROBE_INTERVAL:-0}}"
TEXT_ONLY_MODALITY_SCOPE="${text_only_modality_scope:-${TEXT_ONLY_MODALITY_SCOPE:-audio}}"
LOAD_FROM_CACHE_FILE="${load_from_cache_file:-${LOAD_FROM_CACHE_FILE:-true}}"
SAVE_INTERVAL="${save_interval:-${SAVE_INTERVAL:-400}}"
LOG_INTERVAL="${log_interval:-${LOG_INTERVAL:-1}}"
NUM_WORKERS="${num_workers:-${NUM_WORKERS:-2}}"
DATASET_NUM_PROC="${dataset_num_proc:-${DATASET_NUM_PROC:-2}}"
MOE_AUX_LOSS_COEFF="${moe_aux_loss_coeff:-${MOE_AUX_LOSS_COEFF:-1e-4}}"
# MOE_AUX_LOSS_COEFF="${moe_aux_loss_coeff:-${MOE_AUX_LOSS_COEFF:-0}}"
RECOMPUTE_NUM_LAYERS="${recompute_num_layers:-${RECOMPUTE_NUM_LAYERS:-1}}"
# RECOMPUTE_GRANULARITY="${recompute_granularity:-${RECOMPUTE_GRANULARITY:-full}}"
RECOMPUTE_GRANULARITY="${recompute_granularity:-${RECOMPUTE_GRANULARITY:-selective}}"
MAX_RESAMPLE_TIMES="${max_resample_times:-${MAX_RESAMPLE_TIMES:-1}}"
OPTIMIZER_OFFLOAD_FRACTION="${optimizer_offload_fraction:-${OPTIMIZER_OFFLOAD_FRACTION:-1}}"
NO_SAVE_OPTIM="${no_save_optim:-${NO_SAVE_OPTIM:-true}}"
NO_SAVE_RNG="${no_save_rng:-${NO_SAVE_RNG:-true}}"
EVAL_INTERVAL="${eval_interval:-${EVAL_INTERVAL:-200}}"
TRAIN_ITERS="${train_iters:-${TRAIN_ITERS:-800}}"
REPETITION_PENALTY="${repetition_penalty:-${REPETITION_PENALTY:-1.2}}"
MERGE_LORA="${merge_lora:-${MERGE_LORA:-false}}"
ADAPTERS="${adapters:-${ADAPTERS:-}}"
TEXT_REF_MODEL="${text_ref_model:-${TEXT_REF_MODEL:-}}"
TEXT_REF_LOAD="${text_ref_load:-${TEXT_REF_LOAD:-}}"
REF_MODEL="${ref_model:-${REF_MODEL:-}}"
REWARD_FUNCS="${reward_funcs:-${REWARD_FUNCS:-external_mcqa_accuracy external_format external_consistency}}"
REWARD_WEIGHTS="${reward_weights:-${REWARD_WEIGHTS:-2 1 1}}"
# REWARD_FUNCS="${reward_funcs:-${REWARD_FUNCS:-external_mcqa_accuracy external_format}}"
# REWARD_WEIGHTS="${reward_weights:-${REWARD_WEIGHTS:-1 1}}"

read -r -a REWARD_FUNCS_ARR <<< "${REWARD_FUNCS}"
read -r -a REWARD_WEIGHTS_ARR <<< "${REWARD_WEIGHTS}"

if [[ "${#REWARD_FUNCS_ARR[@]}" -ne "${#REWARD_WEIGHTS_ARR[@]}" ]]; then
    echo "reward_funcs count (${#REWARD_FUNCS_ARR[@]}) must match reward_weights count (${#REWARD_WEIGHTS_ARR[@]})." >&2
    echo "Use quoted strings, for example: --reward-weights \"1 1 1\"" >&2
    exit 1
fi

# Consistency checker (external vLLM server)
CHECKER_PORT="${checker_port:-${CHECKER_PORT:-9000}}"
CHECKER_MODEL_NAME="${checker_model_name:-${CHECKER_MODEL_NAME:-${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Instruct-2507}}"
CHECKER_BASE_URL="${checker_base_url:-${CHECKER_BASE_URL:-http://${ROLLOUT_NODE:-127.0.0.1}:${CHECKER_PORT}/v1}}"
MAPO_CONSISTENCY_CONDITIONAL_ON_CORRECT="${mapo_consistency_conditional_on_correct:-${MAPO_CONSISTENCY_CONDITIONAL_ON_CORRECT:-false}}"
export CHECKER_BASE_URL CHECKER_MODEL_NAME CHECKER_PORT MAPO_CONSISTENCY_CONDITIONAL_ON_CORRECT

CMD=(
    megatron rlhf
    --rlhf_type mapo
    --model "$MODEL_PATH"
    --external_plugins "$PLUGIN_PATH"
    --tuner_type "$TUNER_TYPE"
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --target_modules "$TARGET_MODULES"
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "$ROLLOUT_NODE"
    --vllm_server_port "$VLLM_SERVER_PORT"
    --vllm_server_group_port "$VLLM_SERVER_GROUP_PORT"
    --dataset "$DATASET_PATH"
    --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --micro_batch_size "$MICRO_BATCH_SIZE"
    --global_batch_size "$GLOBAL_BATCH_SIZE_VAL"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --lr "$LR"
    --lr_warmup_fraction "$LR_WARMUP_FRACTION"
    --beta "$BETA"
    --eta "$ETA"
    --mapo_advantage_floor_eps "$MAPO_ADVANTAGE_FLOOR_EPS"
    --mapo_task_fail_gate_floor "$MAPO_TASK_FAIL_GATE_FLOOR"
    --mapo_attn_prefactor_clip "$MAPO_ATTN_PREFACTOR_CLIP"
    --mapo_mask_temperature "$MAPO_MASK_TEMPERATURE"
    --mapo_mask_clip "$MAPO_MASK_CLIP"
    --mapo_temporal_kappa "$MAPO_TEMPORAL_KAPPA"
    --mapo_attention_layers "$MAPO_ATTENTION_LAYERS"
    --mapo_attention_head_reduce "$MAPO_ATTENTION_HEAD_REDUCE"
    --mapo_attention_layer_reduce "$MAPO_ATTENTION_LAYER_REDUCE"
    --mapo_failure_reward_name "$MAPO_FAILURE_REWARD_NAME"
    --mapo_failure_threshold "$MAPO_FAILURE_THRESHOLD"
    --mapo_pos_tags "$MAPO_POS_TAGS"
    --mapo_attention_only "$MAPO_ATTENTION_ONLY"
    --mapo_debug_attn_grad_probe "$MAPO_DEBUG_ATTN_GRAD_PROBE"
    --mapo_debug_attn_grad_probe_interval "$MAPO_DEBUG_ATTN_GRAD_PROBE_INTERVAL"
    --text_only_modality_scope "$TEXT_ONLY_MODALITY_SCOPE"
    --save_interval "$SAVE_INTERVAL"
    --save "$EXP_DIR"
    --log_interval "$LOG_INTERVAL"
    --num_workers "$NUM_WORKERS"
    --dataset_num_proc "$DATASET_NUM_PROC"
    --num_generations "$NUM_GENERATIONS"
    --tensor_model_parallel_size "$TP"
    --pipeline_model_parallel_size "$PP"
    --context_parallel_size "$CP"
    --expert_model_parallel_size "$EP"
    --sequence_parallel true
    --moe_permute_fusion true
    --moe_grouped_gemm true
    --moe_shared_expert_overlap true
    --moe_aux_loss_coeff "$MOE_AUX_LOSS_COEFF"
    --recompute_granularity "$RECOMPUTE_GRANULARITY"
    --vit_gradient_checkpointing true
    --finetune true
    --freeze_vit false
    --freeze_aligner false
    --load_safetensors true
    --save_safetensors true
    --merge_lora "$MERGE_LORA"
    --bf16 true
    --attention_backend flash
    --offload_model true
    --offload_bridge false
    --optimizer_cpu_offload true
    --use_precision_aware_optimizer true
    --optimizer_offload_fraction "$OPTIMIZER_OFFLOAD_FRACTION"
    --log_completions true
    --no_save_optim "$NO_SAVE_OPTIM"
    --no_save_rng "$NO_SAVE_RNG"
    --eval_interval "$EVAL_INTERVAL"
    --system "$SYSTEM_PROMPT"
    --train-iters "$TRAIN_ITERS"
    --temperature "1.0"
    --top_p "0.99"
    --top_k "50"
    --repetition_penalty "$REPETITION_PENALTY"
    # --dynamic-sample true
    # --max-resample-times "$MAX_RESAMPLE_TIMES"
    --overlong-filter true
)

CMD+=(--reward_funcs "${REWARD_FUNCS_ARR[@]}")
CMD+=(--reward_weights "${REWARD_WEIGHTS_ARR[@]}")

if [[ -n "${TEMPLATE}" ]]; then
    CMD+=(--template "${TEMPLATE}")
fi

if [[ "${RECOMPUTE_GRANULARITY}" == "full" ]]; then
    CMD+=(--recompute_method uniform)
    CMD+=(--recompute_num_layers "${RECOMPUTE_NUM_LAYERS}")
fi

if [[ -n "${TEXT_REF_MODEL}" ]]; then
    CMD+=(--text_ref_model "${TEXT_REF_MODEL}")
fi
if [[ -n "${TEXT_REF_LOAD}" ]]; then
    CMD+=(--text_ref_load "${TEXT_REF_LOAD}")
fi
if [[ -n "${REF_MODEL}" ]]; then
    CMD+=(--ref_model "${REF_MODEL}")
fi
if [[ -n "${ADAPTERS}" ]]; then
    CMD+=(--adapters "${ADAPTERS}")
fi

TRAIN_LOG="${LOG_DIR}/train_node${NODE_RANK}_${TIMESTAMP}.log"
echo "Saving training log to: $TRAIN_LOG"
echo "Saving NCCL logs to: ${NCCL_DEBUG_FILE}"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${CMD[@]}" 2>&1 | tee "$TRAIN_LOG" &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_EXIT_CODE=$?
echo "Training finished with exit code ${TRAIN_EXIT_CODE}."
exit ${TRAIN_EXIT_CODE}
