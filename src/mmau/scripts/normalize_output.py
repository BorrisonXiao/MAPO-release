#!/usr/bin/env python

import argparse
import json
import copy
import re
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams


REASONING_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning)\s*>.*?</\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
REASONING_OPEN_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", flags=re.IGNORECASE)
REASONING_CLOSE_RE = re.compile(r"</\s*(think|thinking|reasoning)\s*>", flags=re.IGNORECASE)
ANSWER_TAG_RE = re.compile(
    r"<\s*answer\s*>(.*?)</\s*answer\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
LEADING_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


def strip_reasoning_blocks(text):
    """Remove <think>/<thinking>/<reasoning> blocks from model output."""
    return REASONING_BLOCK_RE.sub("", text).strip()


def extract_answer_tag(text):
    match = ANSWER_TAG_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def extract_post_think_answer(text):
    close_matches = list(REASONING_CLOSE_RE.finditer(text))
    if close_matches:
        trailing_text = text[close_matches[-1].end():].strip()
        if trailing_text:
            return trailing_text

    answer_text = extract_answer_tag(text)
    if answer_text:
        return answer_text

    return strip_reasoning_blocks(text)


def clean_extracted_answer(text):
    cleaned = str(text or "").strip()
    cleaned = LEADING_ANSWER_PREFIX_RE.sub("", cleaned).strip()

    bracket_match = re.fullmatch(
        r"\[\s*(.*?)\s*\]\s*[\.\!\?,;:]*",
        cleaned,
        flags=re.DOTALL,
    )
    if bracket_match:
        cleaned = bracket_match.group(1).strip()

    return cleaned


def normalize_for_match(text):
    normalized = clean_extracted_answer(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"', "`"}:
        normalized = normalized[1:-1].strip()

    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if normalized.startswith(opener) and normalized.endswith(closer):
            normalized = normalized[1:-1].strip()

    normalized = normalized.rstrip(".!?,;:")
    return normalized.casefold()


def match_choice(raw_response, choices):
    normalized_response = normalize_for_match(raw_response)
    if not normalized_response:
        return None
    return next(
        (choice for choice in choices if normalized_response == normalize_for_match(choice)),
        None,
    )


def resolve_answer_extraction_mode(raw_response, configured_mode):
    if configured_mode != "auto":
        return configured_mode

    if ANSWER_TAG_RE.search(raw_response):
        return "answer_tag"
    if REASONING_CLOSE_RE.search(raw_response) or REASONING_OPEN_RE.search(raw_response):
        return "post_think"
    return "answer_tag"


def preprocess_response(raw_response, do_strip_thinking, answer_extraction_mode):
    raw_response = str(raw_response or "").strip()
    if not do_strip_thinking:
        return raw_response

    resolved_mode = resolve_answer_extraction_mode(raw_response, answer_extraction_mode)
    if resolved_mode == "post_think":
        extracted_response = extract_post_think_answer(raw_response)
    else:
        extracted_response = extract_answer_tag(raw_response) or strip_reasoning_blocks(raw_response)

    return clean_extracted_answer(extracted_response)



PROMPT_CLEAN_ANSWER = """
You are given a question, a list of choices, and a model-generated answer (which may be vague or ambiguous).
Your job is to identify and return the exact text of the selected choice.
Respond with only one of the provided options verbatim.

If the answer includes a letter like "A", "B", etc., match it to the corresponding choice.
If the answer text is slightly different (e.g., "A train passing over tracks" vs. "Train passing over tracks"), match it to the closest choice.
If the choices do contain some sort of index but the answer doesn't (e.g., "1. Construction work using power tools" vs. "Construction work using power tools"), match it to the choice, i.e. the one with index.

Only respond with the **exact** matching string from the provided choices.

### Example 1:
Question: "Based on the given audio, identify the source of the speaking voice."
Choices:
["Man", "Woman", "Child", "Robot"]
Model Answer: "The answer is A."
Normalized Answer: Man

### Example 2:
Question: "What is the sound in the audio?"
Choices: ["Train passing over tracks", "Car starting", "Helicopter", "Church bell"]
Model Answer: "A train passing over tracks"
Normalized Answer: Train passing over tracks

### Example 3:
Question: "Based on the given audio, what activity are the men most likely engaged in?"
Choices: ["1. Construction work using power tools", "2. Cooking a meal in the kitchen", "3. Playing a board game", "4. Reading books in a library"]
Model Answer: "Construction work using power tools"
Normalized Answer: 1. Construction work using power tools

However, if the model answer does not clearly correspond to any of the provided choices, respond with an "No answer" string.

Question:
{question}

Choices:
{choices}

Model Answer:
{model_answer}

Normalized Answer:
"""

def load_json_file(file_path: Path):
    """
    Helper to load a standard JSON list. 
    Adjust this if your input is JSON Lines (jsonl).
    """
    with open(file_path, "r") as f:
        # Try loading as a standard JSON list
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Fallback for JSON Lines if needed
            f.seek(0)
            return [json.loads(line) for line in f]

def inference(
    input_file: Path,
    reference_file: Path,
    output_file: Path,
    model_name: str,
    tensor_parallel_size: int = 1,
    output_key: str = "model_prediction",
    do_strip_thinking: bool = False,
    answer_extraction_mode: str = "answer_tag",
):
    # 1. Load Data
    print(f"Loading model outputs from {input_file}...")
    model_data = load_json_file(input_file)
    
    print(f"Loading reference data from {reference_file}...")
    ref_data = load_json_file(reference_file)

    if len(model_data) != len(ref_data):
        print(f"[WARNING] File length mismatch! Input: {len(model_data)}, Reference: {len(ref_data)}")
        print("Processing only the overlapping items based on index alignment.")

    # Prepare storage
    # We will assume row-wise alignment (Item 0 in input matches Item 0 in reference)
    results = [None] * min(len(model_data), len(ref_data))
    items_to_normalize = []
    indices_to_normalize = []

    print("Preprocessing and checking for exact matches...")
    
    # 2. First Pass: Merge, Check Exact Matches, Prepare LLM Prompts
    # We iterate up to the length of the shorter list
    limit = min(len(model_data), len(ref_data))
    
    for i in tqdm(range(limit)):
        model_item = model_data[i]
        ref_item = ref_data[i]
        
        # EXTRACT: Grab the string directly from "response"
        raw_response = preprocess_response(
            model_item.get("response", ""),
            do_strip_thinking=do_strip_thinking,
            answer_extraction_mode=answer_extraction_mode,
        )

        question = ref_item.get("question", "")
        choices = ref_item.get("choices", [])

        # Start building the result object based on the REFERENCE item
        res = copy.deepcopy(ref_item)
        
        # Check for exact case-insensitive match
        matched_choice = match_choice(raw_response, choices)

        if matched_choice:
            # Fast path: No LLM needed
            res[output_key] = matched_choice
            results[i] = res
        else:
            # Slow path: Prepare for vLLM batching
            prompt_content = PROMPT_CLEAN_ANSWER.format(
                question=question,
                choices=choices,
                model_answer=raw_response,
            )
            # Store as a chat message list for the tokenizer
            items_to_normalize.append([{"role": "user", "content": prompt_content}])
            indices_to_normalize.append(i)
            # Placeholder for now
            results[i] = res 

    # 3. Second Pass: Run vLLM on the "hard" cases
    if items_to_normalize:
        print(f"\nInitializing vLLM with model: {model_name}")
        print(f"Tensor Parallel Size: {tensor_parallel_size}")
        
        try:
            llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
            tokenizer = llm.get_tokenizer()
            
            print("Applying chat templates...")
            prompts = [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in items_to_normalize
            ]

            sampling_params = SamplingParams(temperature=0, max_tokens=128)

            print(f"Running inference on {len(prompts)} items...")
            outputs = llm.generate(prompts, sampling_params)

            # Map outputs back to results
            for idx, output in zip(indices_to_normalize, outputs):
                generated_text = output.outputs[0].text.strip()
                # Update the result object that was waiting
                results[idx][output_key] = generated_text

        except Exception as e:
            print(f"\n[ERROR] vLLM Inference failed: {e}")
            return

    # 4. Save Results
    final_results = [r for r in results if r is not None]
    
    with open(output_file, "w") as fout:
        json.dump(final_results, fout, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(final_results)} results to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Standardize answers from model predictions using vLLM."
    )
    parser.add_argument(
        "-i", "--input-file", type=Path, required=True, 
        help="Path to the model output file (contains 'response')"
    )
    parser.add_argument(
        "-r", "--reference-file", type=Path, required=True, 
        help="Path to the original data file (contains 'question' and 'choices')"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default="output.std.json",
        help="Output file",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "-k",
        "--output-key",
        type=str,
        default="model_prediction",
        help="Key for standardized answer in output JSON",
    )
    parser.add_argument(
        "-tp", "--tensor-parallel-size", type=int, default=1, 
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--strip-thinking", action="store_true", default=False,
        help="Strip reasoning blocks before normalization."
    )
    parser.add_argument(
        "--answer-extraction-mode",
        choices=["answer_tag", "post_think", "auto"],
        default="answer_tag",
        help=(
            "How to extract the final answer when --strip-thinking is enabled: "
            "'answer_tag' expects <answer>...</answer>, "
            "'post_think' uses text after </think>/<thinking>/<reasoning>, "
            "and 'auto' infers from the response content."
        ),
    )
    args = parser.parse_args()

    inference(
        input_file=args.input_file,
        reference_file=args.reference_file,
        output_file=args.output_file,
        model_name=args.model_name,
        output_key=args.output_key,
        tensor_parallel_size=args.tensor_parallel_size,
        do_strip_thinking=args.strip_thinking,
        answer_extraction_mode=args.answer_extraction_mode,
    )

if __name__ == "__main__":
    main()
