#!/usr/bin/env python
import argparse
import json
from tqdm import tqdm
from pathlib import Path
import re


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r"\b\w+\b", text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2


def strip_thinking(text):
    """Remove <thinking>...</thinking> blocks from model output."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def extract_prediction(output_str):
    # Step 0: Strip thinking blocks if present
    output_str = strip_thinking(output_str)
    # Step 1: Extract the answer
    extracted = extract_answer(output_str)
    # Step 2: If the extracted answer is the same, i.e. it does not contain <answer> tags, extract the <response></response> part
    if extracted == output_str:
        response_pattern = r"<response>(.*?)</response>"
        match = re.search(response_pattern, output_str, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
    # Step 4: If the extracted answer is still the same, return it as is. Otherwise, return the extracted answer
    return extracted


def extract_answer(output_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, output_str, re.DOTALL)

    if match:
        return match.group(1).strip()
    return output_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process benchmark JSON and calculate accuracy."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file to be evaluated",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file to save results (optional)",
    )
    parser.add_argument(
        "-k",
        "--output-key",
        default="model_prediction",
        type=str,
        help="Key in the JSON to look for model outputs (default: 'model_prediction')",
    )
    parser.add_argument(
        "-e",
        "--exclude-ids",
        type=Path,
        default=None,
        help="Path to a JSON file containing IDs to exclude from evaluation (optional)",
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    if args.exclude_ids:
        with open(args.exclude_ids, "r") as f:
            exclude_ids = set(json.load(f))
        input_data = [item for item in input_data if item["id"] not in exclude_ids]

    corr, total = 0, 0

    # Track metrics for different categories:
    task_metrics = {"sound": [0, 0], "music": [0, 0], "speech": [0, 0]}
    diff_metrics = {"easy": [0, 0], "hard": [0, 0], "medium": [0, 0]}

    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    no_pred_count = 0
    matched_outputs = []
    new_data = []

    output_key = args.output_key

    for idx, sample in enumerate(tqdm(input_data)):
        # If there's no model output key, skip
        if output_key not in sample:
            continue

        if output_key not in sample:
            _prediction = ""
            no_pred_count += 1
        else:
            out = sample[output_key].lower()
            _prediction = extract_prediction(out)

        _answer = extract_answer(sample["answer"].lower())
        task = sample["task"]
        difficulty = sample["difficulty"]
        choices = sample["choices"]

        # Get the sub-category
        subcat = sample.get("sub-category", None)
        if subcat is not None:
            # If we haven't seen this sub-category before, initialize
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample["match"] = 1
        else:
            sample["match"] = 0

        total += 1
        new_data.append(sample)
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1

    # Print results:
    print("*" * 30)
    print("Task-wise Accuracy:")
    task_res = {
        k: v[0] / v[1] * 100 if v[1] > 0 else 0 for k, v in task_metrics.items()
    }
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{task} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    diff_res = {
        k: v[0] / v[1] * 100 if v[1] > 0 else 0 for k, v in diff_metrics.items()
    }
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{diff} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    subcat_res = {
        k: v[0] / v[1] * 100 if v[1] > 0 else 0 for k, v in subcat_metrics.items()
    }
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    print("*" * 30)
    total_res = (corr / total) * 100
    print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    print("*" * 30)
    print(f"No prediction count: {no_pred_count}")

    if args.output:
        print(f"Saving scores to {args.output}")
        with open(args.output, "w") as f:
            json.dump(
                {
                    "task_metrics": task_res,
                    "difficulty_metrics": diff_res,
                    "subcat_metrics": subcat_res,
                    "total_accuracy": total_res,
                    "no_prediction_count": no_pred_count,
                },
                f,
                indent=4,
            )
