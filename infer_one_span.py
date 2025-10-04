import json
import sys

import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from collections import defaultdict

MODEL_PATH_BASE = "distilbert-single-type-simplified"
MARKER_TYPES_TO_INFER = ["Action", "Actor", "Effect", "Evidence", "Victim"]
TEST_FILE = "dev_rehydrated.jsonl"
SUBMISSION_FILE = "submission.jsonl"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 64


def find_latest_checkpoint(base_path, marker_type):
    """
    Scans the type-specific base_path directory for the latest numbered 'checkpoint-*' subfolder.
    """
    full_path = f"{base_path}-{marker_type}"
    checkpoint_dirs = glob.glob(os.path.join(full_path, "checkpoint-*"))

    if not checkpoint_dirs:
        print(f"Warning: No 'checkpoint-*' folder found. Assuming model files are in: {full_path}")
        return full_path

    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))

    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_data(file_path):
    """Loads all data from a JSONL file, preserving order, and retaining the unique ID."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")

                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    """Tokenizes the text and returns inputs with offset mapping for post-processing."""
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128,
                                 return_offsets_mapping=True)

    # We generate dummy labels as the Trainer expects a 'labels' key in the dataset,
    # but the actual values (-100) are ignored during prediction.
    tokenized_inputs["labels"] = [[-100] * len(offset_map) for offset_map in tokenized_inputs["offset_mapping"]]

    return tokenized_inputs


def reconstruct_spans(predictions, tokenized_dataset, id_to_label):
    """
    Converts token-level predictions (O vs TYPE) back into a list of character spans,
    designed for the simplified binary classification model (O or TYPE, no IOB tags).

    Args:
        predictions (np.array): Model output array of predicted label IDs (shape: N samples x M tokens).
        tokenized_dataset (Dataset): The Hugging Face dataset containing 'offset_mapping' and 'text'.
        id_to_label (dict): Mapping from numerical ID to simplified label string (e.g., 1 -> 'EVIDENCE').

    Returns:
        dict: A dictionary mapping sample index to a list of reconstructed marker dictionaries
              for the single marker type inferred.
    """
    reconstructed_markers = defaultdict(list)

    # Determine the positive label type (the one that is not 'O').
    # Assuming ID 0 is 'O' and ID 1 is the marker type.
    positive_label_type = id_to_label.get(1)
    if not positive_label_type or positive_label_type == "O":
        print("Error: Model configuration does not match simplified binary training (ID 1 is not the marker type).")
        return reconstructed_markers

    for i, pred_ids in enumerate(predictions):
        offsets = tokenized_dataset[i]['offset_mapping']
        original_text = tokenized_dataset[i]['text']

        current_span_start_char = None

        # Iterate over tokens
        for token_idx, label_id in enumerate(pred_ids):
            offset_tuple = offsets[token_idx]

            # Check for special tokens, padding, or tokens outside the text range
            is_special = (offset_tuple is None or offset_tuple[0] is None or offset_tuple[1] is None or (
                        offset_tuple[0] == 0 and offset_tuple[1] == 0))

            if is_special:
                # If we were tracking a span, close it using the end of the *previous* non-special token
                if current_span_start_char is not None:
                    prev_end_char = None
                    # Find the end of the last valid token
                    if token_idx > 0 and offsets[token_idx - 1][1] is not None:
                        prev_end_char = offsets[token_idx - 1][1]

                    if prev_end_char is not None:
                        span_text = original_text[current_span_start_char:prev_end_char]
                        reconstructed_markers[i].append({
                            "startIndex": current_span_start_char,
                            "endIndex": prev_end_char,
                            "type": positive_label_type,
                            "text": span_text
                        })

                    current_span_start_char = None
                continue

            label = id_to_label[label_id]
            start_char = offset_tuple[0]

            if label == positive_label_type:
                # Start or continue a span
                if current_span_start_char is None:
                    # Start new span
                    current_span_start_char = start_char

            elif label == 'O':
                # End the span if one was active
                if current_span_start_char is not None:
                    # End is the end of the PREVIOUS token. The token at current_idx is 'O'.
                    # We need the end of the token at token_idx - 1.
                    prev_end_char = offsets[token_idx - 1][1] if token_idx > 0 and offsets[token_idx - 1][
                        1] is not None else start_char

                    span_text = original_text[current_span_start_char:prev_end_char]
                    reconstructed_markers[i].append({
                        "startIndex": current_span_start_char,
                        "endIndex": prev_end_char,
                        "type": positive_label_type,
                        "text": span_text
                    })
                    current_span_start_char = None

        # After loop: Finalize any span that was still open at the end of the sequence
        if current_span_start_char is not None:
            # The end of the span is the end of the last non-special token in the sequence
            last_valid_end = None
            last_token_idx = len(pred_ids) - 1
            # Search backwards from the end of the sequence for the last non-special token's end index
            while last_token_idx >= 0:
                offset_tuple_end = offsets[last_token_idx]
                # Check if the token at this index is not a special token
                if offset_tuple_end is not None and offset_tuple_end[1] is not None and offset_tuple_end[1] != 0:
                    last_valid_end = offset_tuple_end[1]
                    break
                last_token_idx -= 1

            if last_valid_end is not None:
                span_text = original_text[current_span_start_char:last_valid_end]
                reconstructed_markers[i].append({
                    "startIndex": current_span_start_char,
                    "endIndex": last_valid_end,
                    "type": positive_label_type,
                    "text": span_text
                })

    return reconstructed_markers


if __name__ == '__main__':

    # 1. Load Data
    raw_data = load_data(TEST_FILE)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]

    test_dataset = Dataset.from_list(raw_data)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    dummy_label_to_id = {"O": 0}

    tokenized_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if
                        col not in ['text', 'offset_mapping', '_id', 'conspiracy']],
        fn_kwargs={"tokenizer": tokenizer, "label_to_id": dummy_label_to_id}
    )

    # Dictionary to aggregate all predicted markers across all types
    # Map index (0 to len(raw_data) - 1) to a list of markers
    all_predicted_markers = defaultdict(list)

    # 2. Iterate and Infer for Each Marker Type
    for marker_type in MARKER_TYPES_TO_INFER:

        # The new find_latest_checkpoint uses the marker_type to build the full path
        model_directory = find_latest_checkpoint(MODEL_PATH_BASE, marker_type)

        print(f"\n--- Running inference for type: {marker_type} ---")
        print(f"Loading model from: {model_directory}")

        try:
            # Load the model.
            model = DistilBertForTokenClassification.from_pretrained(model_directory)

            # Manually define the id_to_label mapping for the binary model (0=O, 1=TYPE)
            # This ensures the correct marker type name is used instead of generic 'LABEL_1'.
            id_to_label = {0: "O", 1: marker_type}
            print(f"Using defined labels for reconstruction: {id_to_label}")

        except Exception as e:
            print(f"Error loading model for {marker_type} from '{model_directory}'. Details: {e}")
            continue

        # 3. Prepare for Inference using Trainer
        data_collator = DataCollatorForTokenClassification(tokenizer)

        prediction_args = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_inference_span_{marker_type}",
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="none"
            ),
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        # 4. Perform Inference
        predictions_output = prediction_args.predict(tokenized_test_dataset)

        logits = predictions_output.predictions
        predicted_class_ids = np.argmax(logits, axis=2)

        # 5. Reconstruct Spans for this specific marker type using simplified logic
        print(f"Reconstructing character spans for {marker_type}...")
        current_marker_map = reconstruct_spans(predicted_class_ids, tokenized_test_dataset, id_to_label)

        # 6. Aggregate results (index i corresponds to the sample index)
        for i, markers in current_marker_map.items():
            all_predicted_markers[i].extend(markers)

    print(f"\nSaving final aggregated predictions ({len(raw_data)} samples) to {SUBMISSION_FILE} (JSONL format)...")

    jsonl_lines = []
    for i in range(len(raw_data)):
        # Get the predicted markers (aggregated from all models)
        predicted_markers = all_predicted_markers.get(i, [])

        # Create the submission object
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": conspiracy_keys[i],
            "markers": predicted_markers
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")
