import argparse
import json
import re
from pathlib import Path

def extract_choices_from_item(item):
    """
    Extracts choices from an item dictionary.
    Handles 'choice_a', 'choice_b', etc., or a 'choices' list.
    """
    if "choices" in item and isinstance(item["choices"], list):
        return item["choices"]
    
    choices = []
    # Check for keys starting with 'choice_' followed by a single letter
    suffixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for suffix in suffixes:
        key = f"choice_{suffix}"
        if key in item:
            choices.append(item[key])
            
    return choices

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        if not isinstance(text, str):
            return set()
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2

def main():
    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output results file')

    args = parser.parse_args()  
    
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        try:
            input_data = json.load(f)
            if isinstance(input_data, dict):
                input_data = [input_data]
        except json.JSONDecodeError:
            f.seek(0)
            input_data = [json.loads(line) for line in f]

    corr, total = 0, 0
    no_pred_count = 0

    # Dynamic metric containers
    category_metrics = {}
    subcat_metrics = {}
    subsubcat_metrics = {}

    output_key = 'model_prediction'
    
    # We will write the processed data (with 'match' flags) to a new list if you want to save JSON later
    # (The current script writes a text report, but keeping data structure is good practice)
    processed_data = []

    for idx, sample in enumerate(input_data):
        
        # 1. Extract Prediction
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]
            if _prediction is None:
                _prediction = ''
                no_pred_count += 1

        # 2. Extract Metadata
        _answer = sample.get('answer_gt', "")
        
        # Use .get() with defaults for optional fields
        category = sample.get('category', 'Unknown')
        subcat = sample.get('sub-category', 'Unknown')
        subsubcat = sample.get('sub-sub-category', 'Unknown')
        
        choices = extract_choices_from_item(sample)
        
        # Initialize metrics keys if they don't exist
        if category not in category_metrics: category_metrics[category] = [0, 0]
        if subcat not in subcat_metrics: subcat_metrics[subcat] = [0, 0]
        if subsubcat not in subsubcat_metrics: subsubcat_metrics[subsubcat] = [0, 0]

        # 3. Run Matching
        match_result = string_match(_answer, _prediction, choices)

        # 4. Update Counts (Total)
        category_metrics[category][1] += 1
        subcat_metrics[subcat][1] += 1
        subsubcat_metrics[subsubcat][1] += 1
        total += 1

        if match_result:
            # Update Counts (Correct)
            category_metrics[category][0] += 1
            subcat_metrics[subcat][0] += 1
            subsubcat_metrics[subsubcat][0] += 1
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        processed_data.append(sample)

    # Helper to print metrics
    def print_metrics(title, metrics_dict, file_handle=None):
        header = f"*"*30 + "\n" + f"{title}:"
        if file_handle:
            print(header, file=file_handle)
        else:
            print(header)
            
        sorted_keys = sorted(metrics_dict.keys())
        
        for key in sorted_keys:
            n_correct, n_total = metrics_dict[key]
            acc = (n_correct / n_total) * 100 if n_total > 0 else 0
            msg = f"{key} : {acc:.2f}% over {n_total} samples"
            
            if file_handle:
                print(msg, file=file_handle)
            else:
                print(msg)

    # --- Console Output ---
    print_metrics("Category-wise Accuracy", category_metrics)
    print_metrics("Sub-category-wise Accuracy", subcat_metrics)
    print_metrics("Sub-sub-category-wise Accuracy", subsubcat_metrics)

    print("*"*30)
    print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    print("*"*30)
    print(f"No prediction count: {no_pred_count}")

    def build_group(metrics_dict):
        group = {}
        for key, (n_correct, n_total) in metrics_dict.items():
            group[key] = {
                "correct": n_correct,
                "total": n_total,
                "accuracy": (n_correct / n_total) * 100 if n_total > 0 else 0.0,
            }
        return group

    report = {
        "summary": {
            "correct": corr,
            "total": total,
            "total_accuracy": (corr / total) * 100 if total > 0 else 0.0,
            "no_prediction_count": no_pred_count,
        },
        "metric_groups": {
            "category": build_group(category_metrics),
            "sub_category": build_group(subcat_metrics),
            "sub_sub_category": build_group(subsubcat_metrics),
        },
    }

    with open(args.output, 'w') as f_out:
        json.dump(report, f_out, indent=2, ensure_ascii=False)
        f_out.write("\n")

if __name__ == "__main__":
    main()
