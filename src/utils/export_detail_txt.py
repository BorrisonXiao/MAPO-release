#!/usr/bin/env python

import argparse
import json
from pathlib import Path


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    return data["data"]
                return [data]
        except json.JSONDecodeError:
            pass

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def first_present(item, keys):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def collect_choices(item):
    choices = item.get("choices")
    if isinstance(choices, list):
        return choices

    choice_keys = sorted(
        [k for k in item.keys() if k.startswith("choice_")],
        key=lambda k: (len(k), k),
    )
    vals = [item[k] for k in choice_keys if item.get(k) not in (None, "")]
    return vals if vals else None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def write_block(f, title, value):
    f.write(f"{title}:\n")
    text = fmt(value)
    if text:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Export a human-readable detail report from raw + normalized outputs.")
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--normalized-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    raw_records = load_records(args.raw_output)
    norm_records = load_records(args.normalized_output)

    raw_by_id = {
        str(r.get("id")): r
        for r in raw_records
        if isinstance(r, dict) and r.get("id") is not None
    }
    use_id = bool(raw_by_id) and all(isinstance(r, dict) and r.get("id") is not None for r in norm_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(norm_records, start=1):
            if not isinstance(item, dict):
                continue

            item_id = item.get("id")
            raw_item = raw_by_id.get(str(item_id)) if use_id else (raw_records[idx - 1] if idx - 1 < len(raw_records) else {})
            if not isinstance(raw_item, dict):
                raw_item = {}

            gt_value = first_present(
                item,
                [
                    "answer_gt",
                    "ground_truth",
                    "ground-truth",
                    "answer",
                    "gt",
                    "label",
                    "correct_answer",
                    "target",
                ],
            )
            extracted_answer = first_present(
                item,
                ["model_prediction", "prediction", "pred", "extracted_answer", "answer_pred"],
            )
            raw_response = raw_item.get("response")
            normalized_original_response = item.get("original_response")
            sub_category_fields = {
                k: v
                for k, v in item.items()
                if "sub" in k.lower() and "categor" in k.lower() and v not in (None, "")
            }

            f.write(f"{'=' * 24} Sample {idx} {'=' * 24}\n\n")
            f.write(f"id: {fmt(item.get('id'))}\n")
            f.write(f"task_name: {fmt(first_present(item, ['task_name', 'task', 'task_type']))}\n")
            f.write(f"audio_path: {fmt(first_present(item, ['audio_path', 'audio', 'wav_path', 'file_path', 'path']))}\n")
            f.write(f"category: {fmt(first_present(item, ['category', 'main_category']))}\n")
            f.write(f"ground_truth: {fmt(gt_value)}\n")
            f.write(f"extracted_answer: {fmt(extracted_answer)}\n\n")

            write_block(f, "question", item.get("question"))
            write_block(f, "choices", collect_choices(item))
            write_block(f, "system_prompt", args.system_prompt)
            write_block(f, "sub_categories", sub_category_fields or first_present(item, ["sub_categories", "sub-categories"]))
            write_block(f, "raw_response", raw_response)
            if normalized_original_response not in (None, "") and normalized_original_response != raw_response:
                write_block(f, "normalized_original_response", normalized_original_response)
            f.write("\n")


if __name__ == "__main__":
    main()
