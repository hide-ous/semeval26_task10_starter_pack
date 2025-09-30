import json
import argparse
import os
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

CATEGORIES = ["No", "Yes"]


def load_jsonl(filepath, id_field="_id", label_field=None):
    """
    Loads data from a JSONL file, returning a dictionary mapping the unique ID
    to the label (or prediction).
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                unique_id = item.get(id_field)
                label = item.get(label_field)

                if unique_id is None:
                    print(f"Warning: Skipping line {i + 1} in {filepath}. Missing '{id_field}'.")
                    continue
                if label is None:
                    print(f"Warning: Skipping line {i + 1} in {filepath}. Missing '{label_field}'.")
                    continue

                if label not in CATEGORIES:
                    print(f"Warning: Skipping line {i + 1} in {filepath}. Invalid label '{label}'.")
                    continue

                data[unique_id] = label
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i + 1} in {filepath}. Skipping.")
                continue
    return data


def evaluate_submission(reference_file, submission_file, output_path="."):
    """
    Loads reference and submission data, matches them by ID, calculates metrics,
    and saves the results to scores.json.
    """
    # 1. Load Data
    ref_data = load_jsonl(reference_file, id_field="_id", label_field="conspiracy")
    sub_data = load_jsonl(submission_file, id_field="_id", label_field="conspiracy")

    if not ref_data or not sub_data:
        print("Evaluation failed: One or both input files are empty or invalid.")
        sys.exit(1)

    # 2. Match Data and Filter
    matched_ids = sorted(list(set(ref_data.keys()) & set(sub_data.keys())))

    if not matched_ids:
        print("Error: No common document IDs found between reference and submission files.")
        sys.exit(1)

    y_true = [ref_data[id] for id in matched_ids]
    y_pred = [sub_data[id] for id in matched_ids]

    print(f"Matched {len(matched_ids)} samples for evaluation.")

    # 3. Calculate Metrics

    # Weighted F1 Score
    f1_weighted = f1_score(y_true, y_pred, labels=CATEGORIES, average='weighted', zero_division=0)

    # Other useful metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0
    )

    # 4. Prepare Scores for Output
    scores = {
              "f1_score_weighted": float(f1_weighted),
              "accuracy": float(accuracy),
              "f1_score_no": float(f1_scores[CATEGORIES.index("No")]),
              "f1_score_yes": float(f1_scores[CATEGORIES.index("Yes")]),
    }

    # 5. Save Scores
    output_filename = os.path.join(output_path, "scores.json")
    os.makedirs(output_path, exist_ok=True)

    with open(output_filename, 'w') as f:
        json.dump(scores, f, indent=4)

    print("-" * 30)
    print(f"Evaluation complete. Results saved to {output_filename}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification predictions.")

    # Changed from positional to optional with a default
    parser.add_argument(
        "--reference-file",
        type=str,
        default="test.jsonl",
        help="Path to the ground truth JSONL file (default: test.jsonl)."
    )
    parser.add_argument(
        "--submission-file",
        default='submission.json',
        type=str,
        help="Path to the submission JSONL file (e.g., submission.jsonl)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save scores.json (default: current directory)."
    )

    args = parser.parse_args()

    evaluate_submission(args.reference_file, args.submission_file, args.output_dir)
