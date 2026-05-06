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
stop_stage=1
suffix=""
result_dir=""
input_name="qwen3-omni"
model_tag="Qwen3-Omni-Thinking"
llm_model_name_or_path="${MODEL_ROOT}/Qwen/Qwen2.5-7B-Instruct"
embed_model_name_or_path="${MODEL_ROOT}/nvidia/NV-Embed-v2"
model_output_column="model_prediction"

. "${script_dir}/../utils/parse_options.sh" || true

if [ -z "${result_dir}" ]; then
    result_dir="${MAPO_ROOT}/exp/mmau-pro/results/Qwen3-Omni-30B-A3B-Thinking"
fi

input_path="${result_dir}/${input_name}${suffix}.normalized.parquet"
output_path="${result_dir}/${input_name}${suffix}.eval.json"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python "${script_dir}/scripts/evaluation.py" \
        "${input_path}" \
        --llm_model_name_or_path "${llm_model_name_or_path}" \
        --embed_model_name_or_path "${embed_model_name_or_path}" \
        --model_output_column "${model_output_column}" \
        -o "${output_path}"

    cd "${result_dir}"
    ln -sfv "$(basename "${output_path}")" "${model_tag}.json"
    cd - >/dev/null
fi
