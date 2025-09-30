import json
import os
import sys
import argparse
import re
from typing import Dict, List, Set, Tuple

# --- Configuration ---
DEFAULT_TEST_FILE = "test.jsonl"
DEFAULT_SUBMISSION_FILE = "submission_span.jsonl"
DEFAULT_SCORES_FILE = "scores.json"

MARKER_TYPES = {"Action", "Actor", "Effect", "Evidence", "Victim"}
DEFAULT_IOU_THRESHOLD = 0.5


# --- Tokenization and Span Conversion ---

def tokenize_text(text: str) -> List[Tuple[int, int]]:
    """
    Performs simple, robust tokenization based on non-whitespace, non-word characters.
    Returns a list of (start_char, end_char) tuples for each token.
    This mimics simple NLP tokenizers without external libraries.
    """
    # Matches words (\w+) or non-whitespace/non-word punctuation ([^\w\s])
    # This ensures punctuation is treated as separate tokens.
    token_spans = []
    # Use re.finditer to get all matches and their start/end indices
    for match in re.finditer(r"(\w+|[^\w\s])", text):
        token_spans.append((match.start(), match.end()))
    return token_spans


def char_span_to_token_set(
        char_start: int,
        char_end: int,
        token_spans: List[Tuple[int, int]]
) -> Set[int]:
    """
    Converts a character span (start, end) into a set of token indices it covers.
    A token is covered if the character span overlaps with the token's character span.
    """
    covered_token_indices = set()

    for token_idx, (t_start, t_end) in enumerate(token_spans):
        # Overlap exists if: (start_A < end_B) AND (end_A > start_B)
        # Check for character overlap between marker span and token span
        if char_start < t_end and char_end > t_start:
            covered_token_indices.add(token_idx)

    return covered_token_indices


def calculate_token_iou(set_a: Set[int], set_b: Set[int]) -> float:
    """Calculates IoU based on the intersection and union of token index sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty sets, perfect match

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    if not union:
        return 0.0

    return len(intersection) / len(union)


# --- Data Handling and Evaluation ---

def parse_args():
    """Parses command-line arguments for file paths and configuration."""
    parser = argparse.ArgumentParser(
        description="Evaluate span extraction predictions against ground truth using TOKEN-BASED Overlap F1 (IoU >= threshold)."
    )

    # Positional Arguments (Codabench style)
    parser.add_argument(
        "--ground_truth_file",
        nargs='?',
        default=DEFAULT_TEST_FILE,
        help="Path to the ground truth (test) JSONL file, which MUST contain the 'text' field. (Default: %(default)s)"
    )
    parser.add_argument(
        "--prediction_file",
        nargs='?',
        default=DEFAULT_SUBMISSION_FILE,
        help="Path to the submission (predicted) JSONL file. (Default: %(default)s)"
    )
    parser.add_argument(
        "--scores_output_file",
        nargs='?',
        default=DEFAULT_SCORES_FILE,
        help="Path to the output JSON file for Codabench scores. (Default: %(default)s)"
    )

    # Optional Flag for Threshold
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="The minimum token IoU required to consider a match a True Positive. (Default: %(default).1f)"
    )

    return parser.parse_args()


def load_jsonl(file_path):
    """Loads all data from a JSONL file, ensuring file existence check."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: Required file not found at {file_path}", file=sys.stderr)
        if len(sys.argv) > 1:
            sys.exit(1)
        return None

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file_path}: {line.strip()}", file=sys.stderr)
    return data


def extract_markers(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Extracts markers from data, mapping _id to a list of mutable marker objects
    with a 'matched' flag.
    """
    prepared = {}
    for item in data:
        doc_id = item.get('_id')
        if not doc_id:
            continue

        markers_list = [
            {'start': m.get('startIndex'), 'end': m.get('endIndex'), 'type': m.get('type'), 'matched': False}
            for m in item.get('markers', [])
            if m.get('type') in MARKER_TYPES and
               isinstance(m.get('startIndex'), int) and
               isinstance(m.get('endIndex'), int) and
               m.get('startIndex') < m.get('endIndex')
        ]
        prepared[doc_id] = markers_list
    return prepared


def prepare_true_data(data: List[Dict]) -> Dict[str, Dict]:
    """
    Prepares ground truth data: tokenizes text and extracts true markers.
    """
    prepared = {}
    for item in data:
        doc_id = item.get('_id')
        text = item.get('text', '')  # Text is MANDATORY for ground truth here
        if not doc_id or not text:
            # Skip documents without text (required for tokenization)
            continue

        # 1. Tokenize the text once per document
        token_spans = tokenize_text(text)

        # 2. Extract and prepare markers
        markers_list = [
            {'start': m.get('startIndex'), 'end': m.get('endIndex'), 'type': m.get('type'), 'matched': False}
            for m in item.get('markers', [])
            if m.get('type') in MARKER_TYPES and
               isinstance(m.get('startIndex'), int) and
               isinstance(m.get('endIndex'), int) and
               m.get('startIndex') < m.get('endIndex')
        ]

        prepared[doc_id] = {
            'token_spans': token_spans,
            'markers': markers_list
        }
    return prepared


def evaluate(true_data, pred_data, iou_threshold):
    """
    Calculates Token-Based Overlap Match F1-Score for spans.
    """
    if true_data is None or pred_data is None:
        return {"Error": "Data loading failed."}

    # Prepare GT data: Tokenize text and prepare true markers
    true_docs = prepare_true_data(true_data)

    # Prepare predicted markers (only character offsets needed)
    pred_markers_map = extract_markers(pred_data)

    # Aggregate counters for all types
    total_tp, total_fp, total_fn = 0, 0, 0
    type_metrics = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in MARKER_TYPES}

    # Only iterate over IDs present in the ground truth set
    all_ids = true_docs.keys()

    for doc_id in all_ids:
        true_doc = true_docs.get(doc_id)

        # If true_doc is missing, it means the GT document was skipped because it lacked 'text',
        # but we shouldn't run logic on it anyway.
        if not true_doc:
            continue

        true_spans = true_doc['markers']
        token_spans = true_doc['token_spans']  # Use ground truth token spans
        pred_spans = pred_markers_map.get(doc_id, [])  # Get predicted spans for this doc

        # --- Matching Loop ---
        for true_span in true_spans:
            true_token_set = char_span_to_token_set(
                true_span['start'], true_span['end'], token_spans
            )

            best_iou = -1.0
            best_pred_idx = -1

            # 1. Find the best *unmatched* predicted span of the same type
            for pred_idx, pred_span in enumerate(pred_spans):
                if pred_span['matched'] or pred_span['type'] != true_span['type']:
                    continue

                # Convert predicted char span to token set using the GT token map
                pred_token_set = char_span_to_token_set(
                    pred_span['start'], pred_span['end'], token_spans
                )

                iou = calculate_token_iou(true_token_set, pred_token_set)

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            # 2. Check if the best match meets the IoU threshold
            if best_iou >= iou_threshold and best_pred_idx != -1:
                # True Positive (TP)
                total_tp += 1
                type_metrics[true_span['type']]['tp'] += 1
                true_span['matched'] = True
                pred_spans[best_pred_idx]['matched'] = True

        # 3. After matching, calculate FP and FN

        # FN: Unmatched true spans
        for true_span in true_spans:
            if not true_span['matched']:
                total_fn += 1
                type_metrics[true_span['type']]['fn'] += 1

        # FP: Unmatched predicted spans
        for pred_span in pred_spans:
            if not pred_span['matched']:
                if pred_span['type'] in MARKER_TYPES:
                    total_fp += 1
                    type_metrics[pred_span['type']]['fp'] += 1

    final_results = {}
    all_f1_scores = []

    # Calculate Per-Type Metrics
    for m_type in sorted(MARKER_TYPES):
        metrics = type_metrics[m_type]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        all_f1_scores.append(f1)

        final_results[f"P ({m_type})"] = precision
        final_results[f"R ({m_type})"] = recall
        final_results[f"F1 ({m_type})"] = f1

    # Calculate Macro F1
    f1_macro = sum(all_f1_scores) / len(MARKER_TYPES) if MARKER_TYPES else 0
    final_results["F1 (Macro)"] = f1_macro

    # Calculate Aggregate (Micro) Metrics
    precision_agg = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall_agg = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_agg = 2 * (precision_agg * recall_agg) / (precision_agg + recall_agg) if (precision_agg + recall_agg) > 0 else 0

    final_results["P (Agg)"] = precision_agg
    final_results["R (Agg)"] = recall_agg
    final_results["F1 (Agg)"] = f1_agg

    # Store formatted strings for console output
    final_results_formatted = {
        "--- Per-Type Results (Token IoU) ---": "---"
    }
    for m_type in sorted(MARKER_TYPES):
        final_results_formatted[f"P ({m_type})"] = f"{final_results[f'P ({m_type})']:.4f}"
        final_results_formatted[f"R ({m_type})"] = f"{final_results[f'R ({m_type})']:.4f}"
        final_results_formatted[f"F1 ({m_type})"] = f"{final_results[f'F1 ({m_type})']:.4f}"

    final_results_formatted["--- Aggregate Results ---"] = "---"
    final_results_formatted["IoU Threshold"] = iou_threshold
    final_results_formatted["True Positives (Agg)"] = total_tp
    final_results_formatted["False Positives (Agg)"] = total_fp
    final_results_formatted["False Negatives (Agg)"] = total_fn
    final_results_formatted["Precision (Agg)"] = f"{precision_agg:.4f}"
    final_results_formatted["Recall (Agg)"] = f"{recall_agg:.4f}"
    final_results_formatted["F1-Score (Agg/Micro)"] = f"{f1_agg:.4f}"
    final_results_formatted["F1-Score (Macro)"] = f"{f1_macro:.4f}"

    return final_results, final_results_formatted


def save_scores_to_codabench(results, output_file):
    """
    Saves the final scores to a JSON file in the format expected by Codabench.
    """
    scores = dict()

    # Save Aggregate (Micro) Scores
    scores["F1_Aggregate_Token"]= results["F1 (Agg)"]
    scores["Precision_Aggregate_Token"]= results["P (Agg)"]
    scores["Recall_Aggregate_Token"]= results["R (Agg)"]

    # Save Macro Score
    scores["F1_Macro_Token"]= results["F1 (Macro)"]

    # Save Per-Type Scores
    for m_type in sorted(MARKER_TYPES):
        scores[f"F1_{m_type}_Token"] = results[f"F1 ({m_type})"]
        scores[f"Precision_{m_type}_Token"] = results[f"P ({m_type})"]
        scores[f"Recall_{m_type}_Token"] = results[f"R ({m_type})"]

    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=4)

    print(f"Token-based scores saved to {output_file} for Codabench compatibility.")


if __name__ == "__main__":

    # 1. Parse Command-Line Arguments
    args = parse_args()

    TEST_FILE = args.ground_truth_file
    SUBMISSION_FILE = args.prediction_file
    SCORES_FILE = args.scores_output_file
    IOU_THRESHOLD_RUNTIME = args.iou_threshold

    print(f"Starting TOKEN-BASED evaluation (Overlap Match F1 with IoU >= {IOU_THRESHOLD_RUNTIME}).")
    print(f"Ground Truth (must contain text): {TEST_FILE}")
    print(f"Predictions (character offsets): {SUBMISSION_FILE}")

    # 2. Load Data
    true_data = load_jsonl(TEST_FILE)
    pred_data = load_jsonl(SUBMISSION_FILE)

    if true_data is None or pred_data is None:
        print("Evaluation terminated due to file loading errors.")
        default_results = {"F1 (Agg)": 0.0, "P (Agg)": 0.0, "R (Agg)": 0.0, "F1 (Macro)": 0.0}
        for m_type in MARKER_TYPES:
            default_results[f"F1 ({m_type})"] = 0.0
            default_results[f"P ({m_type})"] = 0.0
            default_results[f"R ({m_type})"] = 0.0

        save_scores_to_codabench(default_results, SCORES_FILE)
    else:
        raw_results, formatted_results = evaluate(true_data, pred_data, iou_threshold=IOU_THRESHOLD_RUNTIME)

        save_scores_to_codabench(raw_results, SCORES_FILE)

        print("\n--- Token-Based Evaluation Results ---")
        for key, value in formatted_results.items():
            if key.startswith("---"):
                print(key)
            else:
                print(f"{key:<30}: {value}")
        print("--------------------------------------")
