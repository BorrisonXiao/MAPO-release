#!/bin/bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MAPO_ROOT="${MAPO_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(cd -- "${MAPO_ROOT}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/.cache/downloads/model}"

stage=1
stop_stage=3

suffix=""
max_new_tokens=2048
system_prompt=""
model_root="${MODEL_ROOT}"
model_name_or_path="${model_root}/Qwen/Qwen3-Omni-30B-A3B-Thinking"
base_model_name_or_path=""
adapter_name_or_path=""
merge_lora="false"
safe_serialization="true"
max_shard_size="5GB"
result_dir=""
devices="0,1,2,3"
model_tag="Qwen3-Omni-Thinking"
val_dataset="${MAPO_ROOT}/data/raw/mmau/processed-mmau-test-mini.jsonl"
reference_file="${MAPO_ROOT}/data/raw/mmau/mmau-test-mini.json"
output_name="qwen3-omni"
vllm_tensor_parallel_size=""
answer_extraction_mode="auto"
runtime_cache_root=""
runtime_cache_alias_root=""

. "${script_dir}/../utils/parse_options.sh" || true

detail_export_script="${script_dir}/../utils/export_detail_txt.py"
resolver_script="${script_dir}/../utils/resolve_infer_model.py"
answer_extraction_utils="${script_dir}/../utils/answer_extraction_mode.sh"

if [ -z "${vllm_tensor_parallel_size}" ]; then
    IFS=',' read -r -a _mmau_devices <<<"${devices}"
    vllm_tensor_parallel_size=${#_mmau_devices[@]}
fi

if [ -z "${result_dir}" ]; then
    result_dir="${MAPO_ROOT}/exp/mmau/results/Qwen3-Omni-30B-A3B-Thinking/test-mini"
fi

if [ -z "${runtime_cache_root}" ]; then
    runtime_cache_root="${RUNTIME_CACHE_ROOT:-${result_dir}/.runtime-cache}"
fi

mkdir -p "${runtime_cache_root}"

if [ -z "${runtime_cache_alias_root}" ]; then
    runtime_cache_token="$(printf '%s' "${runtime_cache_root}" | cksum | awk '{print $1}')"
    runtime_cache_alias_root="${RUNTIME_CACHE_ALIAS_ROOT:-/tmp/mrt-${runtime_cache_token}}"
fi

mkdir -p "$(dirname "${runtime_cache_alias_root}")"
ln -sfn "${runtime_cache_root}" "${runtime_cache_alias_root}"

short_runtime_cache_root="${runtime_cache_alias_root}"
default_tmpdir="${short_runtime_cache_root}/tmp"
default_vllm_cache_root="${short_runtime_cache_root}/vllm"
default_torchinductor_cache_dir="${short_runtime_cache_root}/torchinductor"
default_triton_cache_dir="${short_runtime_cache_root}/triton"
default_cuda_cache_path="${short_runtime_cache_root}/nv"

mkdir -p \
    "${default_tmpdir}" \
    "${default_vllm_cache_root}" \
    "${default_torchinductor_cache_dir}" \
    "${default_triton_cache_dir}" \
    "${default_cuda_cache_path}"

export TMPDIR="${TMPDIR:-${default_tmpdir}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${default_vllm_cache_root}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${default_torchinductor_cache_dir}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${default_triton_cache_dir}}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${default_cuda_cache_path}}"

log "Runtime cache root: ${runtime_cache_root}"
log "Runtime cache alias: ${short_runtime_cache_root}"
log "TMPDIR=${TMPDIR}"
log "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
log "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
log "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

if [ ! -f "${resolver_script}" ]; then
    log "Resolver script not found: ${resolver_script}"
    exit 1
fi
if [ ! -f "${answer_extraction_utils}" ]; then
    log "Answer extraction helper not found: ${answer_extraction_utils}"
    exit 1
fi

. "${answer_extraction_utils}"

eval "$(
    python "${resolver_script}" \
        --model-name-or-path "${model_name_or_path}" \
        --base-model-name-or-path "${base_model_name_or_path}" \
        --adapter-name-or-path "${adapter_name_or_path}"
)"
model_name_or_path="${RESOLVED_MODEL_NAME_OR_PATH}"
adapter_name_or_path="${RESOLVED_ADAPTER_NAME_OR_PATH}"
adapter_opts=()
if [ -n "${adapter_name_or_path}" ]; then
    adapter_opts=(--adapters "${adapter_name_or_path}")
    log "Using base model: ${model_name_or_path}"
    log "Using adapter: ${adapter_name_or_path}"
fi

if [ "${merge_lora,,}" = "true" ]; then
    if [ -z "${adapter_name_or_path}" ]; then
        log "--merge-lora true requires an adapter checkpoint."
        exit 1
    fi
    if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
        merged_model_name_or_path="${adapter_name_or_path}-merged"
        if [ -d "${merged_model_name_or_path}" ]; then
            log "Found existing merged model, skipping merge: ${merged_model_name_or_path}"
        else
            if [ -e "${merged_model_name_or_path}" ]; then
                log "Merge output path exists but is not a directory: ${merged_model_name_or_path}"
                exit 1
            fi
            log "Pre-merging adapter in a separate process before inference."
            ENABLE_AUDIO_OUTPUT=0 \
                CUDA_VISIBLE_DEVICES=${devices} \
                swift export \
                --model "${model_name_or_path}" \
                --adapters "${adapter_name_or_path}" \
                --merge_lora true \
                --safe_serialization "${safe_serialization}" \
                --max_shard_size "${max_shard_size}" \
                --output_dir "${merged_model_name_or_path}"
        fi
        model_name_or_path="${merged_model_name_or_path}"
        adapter_name_or_path=""
        adapter_opts=()
        log "Using merged model: ${model_name_or_path}"
    fi
fi

resolved_answer_extraction_mode="${answer_extraction_mode}"
if [ "${answer_extraction_mode}" = "auto" ]; then
    resolved_answer_extraction_mode="$(
        resolve_answer_extraction_mode \
            "post_think" \
            "${base_model_name_or_path:-}" \
            "${model_name_or_path:-}" \
            "${adapter_name_or_path:-}" \
            "${model_tag:-}" \
            "${result_dir:-}"
    )"
fi
log "Using answer extraction mode: ${resolved_answer_extraction_mode}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    system_opts=()
    if [ -n "${system_prompt}" ]; then
        system_opts=(--system "${system_prompt}")
    fi

    ENABLE_AUDIO_OUTPUT=0 \
        CUDA_VISIBLE_DEVICES=${devices} \
        swift infer \
        --model ${model_name_or_path} \
        --val_dataset ${val_dataset} \
        --result_path ${result_dir}/${output_name}${suffix}.jsonl \
        --vllm_gpu_memory_utilization 0.75 \
        --infer_backend vllm \
        --vllm_tensor_parallel_size ${vllm_tensor_parallel_size} \
        --vllm_max_model_len 32768 \
        --max_new_tokens ${max_new_tokens} \
        --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}' "${adapter_opts[@]}" "${system_opts[@]}"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${devices} \
        python "${script_dir}/scripts/normalize_output.py" \
        -i ${result_dir}/${output_name}${suffix}.jsonl \
        -o ${result_dir}/${output_name}${suffix}.normalized.jsonl \
        -r ${reference_file} \
        -m ${model_root}/Qwen/Qwen2.5-7B-Instruct \
        --strip-thinking \
        --answer-extraction-mode "${resolved_answer_extraction_mode}"

    python "${detail_export_script}" \
        --raw-output "${result_dir}/${output_name}${suffix}.jsonl" \
        --normalized-output "${result_dir}/${output_name}${suffix}.normalized.jsonl" \
        --output "${result_dir}/${output_name}${suffix}.detail.txt" \
        --system-prompt "${system_prompt}"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    python "${script_dir}/scripts/evaluation.py" \
        -i ${result_dir}/${output_name}${suffix}.normalized.jsonl \
        -o ${result_dir}/${output_name}${suffix}.eval.json

    cd "${result_dir}"
    ln -sfv "${output_name}${suffix}.eval.json" "${model_tag}.json"
    cd - >/dev/null
fi
