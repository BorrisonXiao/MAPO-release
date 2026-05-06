# MAPO

Minimal public export for MAPO training and benchmark inference code.

This tree intentionally keeps only the pieces needed to install the code, launch MAPO training, and run inference on MMAU/MMAR/MMSU/MMAU-Pro:

```text
mapo-release/
|-- requirements.txt
|-- src/
|   |-- mapo/
|   |   |-- start_train.sh
|   |   |-- start_rollout.sh
|   |   `-- start_checker.sh
|   |-- mmau/
|   |   |-- infer.sh
|   |   `-- scripts/
|   |-- mmar/
|   |   |-- infer.sh
|   |   `-- scripts/
|   |-- mmsu/
|   |   |-- infer.sh
|   |   `-- scripts/
|   |-- mmau-pro/
|   |   |-- infer.sh
|   |   |-- eval.sh
|   |   `-- scripts/
|   |-- rewards/
|   |   `-- audio_qa_rewards.py
|   `-- utils/
|       |-- answer_extraction_mode.sh
|       |-- export_detail_txt.py
|       |-- parse_options.sh
|       `-- resolve_infer_model.py
|-- libs/
|   `-- ms-swift/
|       `-- swift/megatron/
|           |-- arguments/megatron_args.py
|           `-- trainers/
|               |-- mapo_attention_collector.py
|               |-- mapo_pos_utils.py
|               `-- mapo_trainer.py
```

## Install

Use Python 3.11. The reference environment used CUDA 12.8, PyTorch 2.9.1, Megatron-Core 0.15.4, and vLLM 0.14.0.

```bash
git clone <public_repo_url> mapo-release
cd mapo-release
```

With `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

With conda:

```bash
conda create -n mapo python=3.11 -y
conda activate mapo
pip install --upgrade pip
pip install -r requirements.txt
```

`start_train.sh` will use `MEGATRON_LM_PATH` if it is set; otherwise it clones NVIDIA Megatron-LM `core_r0.15.0` into `libs/Megatron-LM` on first launch.

## Environment

```bash
export MAPO_ROOT="$(pwd)"
export ROOT_DIR="$(dirname "$MAPO_ROOT")"
export MODEL_ROOT="${ROOT_DIR}/.cache/downloads/model"
export DATA_ROOT="${MAPO_ROOT}/data"
```

Place datasets under `${DATA_ROOT}` or pass `--dataset-path` explicitly. Model weights are not included.

## Launch

Start the rollout server on the rollout node:

```bash
bash "${MAPO_ROOT}/src/mapo/start_rollout.sh" \
  --model-path "${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Thinking" \
  --gpus 4 \
  --tp 4 \
  --vllm-port 8050
```

Optionally start the consistency-checker server:

```bash
bash "${MAPO_ROOT}/src/mapo/start_checker.sh" \
  --model-path "${MODEL_ROOT}/Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --gpus 4 \
  --checker-port 9000
```

Start training on each training node:

```bash
MASTER_ADDR=<master_ip> \
ROLLOUT_NODE=<rollout_ip> \
NODE_RANK=0 \
bash "${MAPO_ROOT}/src/mapo/start_train.sh" \
  --nnodes 3 \
  --gpus 8 \
  --model-path "${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Thinking" \
  --dataset-path "${DATA_ROOT}/train.jsonl" \
  --exp-dir "${MAPO_ROOT}/exp/mapo/run"
```

Run node ranks `1`, `2`, etc. on the remaining training nodes.

## Benchmark Inference

The benchmark launchers share the same stages:

- Stage 1: `swift infer` with vLLM.
- Stage 2: normalize model outputs and write a human-readable detail file.
- Stage 3: compute benchmark accuracy where the benchmark has a local evaluator.

Expected default dataset locations:

```text
data/raw/mmau/processed-mmau-test-mini.jsonl
data/raw/mmau/mmau-test-mini.json
data/raw/mmar/processed-mmar.jsonl
data/raw/mmar/MMAR-meta.json
data/raw/mmsu/processed-mmsu.jsonl
data/raw/mmsu/mmsu.jsonl
data/raw/mmau-pro/processed-mmau-pro.jsonl
data/raw/mmau-pro/mmau-pro.json
```

Run any benchmark with explicit paths:

```bash
bash "${MAPO_ROOT}/src/mmau/infer.sh" \
  --model-name-or-path "${MODEL_ROOT}/Qwen/Qwen3-Omni-30B-A3B-Thinking" \
  --val-dataset "${DATA_ROOT}/raw/mmau/processed-mmau-test-mini.jsonl" \
  --reference-file "${DATA_ROOT}/raw/mmau/mmau-test-mini.json" \
  --result-dir "${MAPO_ROOT}/exp/mmau/results/my-run/test-mini" \
  --devices "0,1,2,3"
```

Swap `mmau` for `mmar`, `mmsu`, or `mmau-pro`. For adapter checkpoints, pass `--adapter-name-or-path`; if the model path itself is an adapter-only checkpoint, the resolver will infer it and use the base model from `args.json` when available. MMAU-Pro writes a normalized parquet file during inference; run `src/mmau-pro/eval.sh` afterward to produce the final evaluation JSON.

## MAPO Arguments

The public launcher is aligned with MAPO v2.5. The release keeps the entropy-weighted policy-gradient mask and the audio-attention loss branch.

- `--eta`: attention-loss branch weight.
- `--mapo-advantage-floor-eps`: floor used when scaling PG weights by advantage magnitude.
- `--mapo-task-fail-gate-floor`: lower bound for the failed-task gate in the attention branch.
- `--mapo-attn-prefactor-clip`: optional upper bound for the per-sequence attention-loss prefactor; `0` disables clipping.
- `--mapo-mask-temperature`, `--mapo-mask-clip`: controls for the PG/attention relevance weights.
- `--mapo-temporal-kappa`: temporal weighting exponent for the attention branch.
- `--mapo-attention-layers`: comma-separated attention layers to collect, or `all`.
- `--mapo-attention-head-reduce`, `--mapo-attention-layer-reduce`: `max` or `mean` aggregation.
- `--mapo-failure-reward-name`, `--mapo-failure-threshold`: reward signal used to gate failed examples.
- `--mapo-pos-tags`: comma-separated POS tags used by the attention branch gate.
- `--mapo-attention-only`: diagnostic mode that optimizes only `eta * attn_loss`.
- `--text-only-modality-scope`: `audio` or `all`.
- `--text-ref-model`, `--text-ref-load`: optional separate text-reference model/checkpoint for entropy-delta weighting.

## License

This project is released under the Apache License 2.0. See `LICENSE`.

The vendored `libs/ms-swift` package carries its original Apache-2.0 license in `libs/ms-swift/LICENSE`. Model weights and datasets are not included in this repository and are governed by their own licenses.
