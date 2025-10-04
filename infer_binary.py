import json
import sys

import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import TrainingArguments

# --- Configuration matching the training script ---
MODEL_PATH = "distilbert-conspiracy-classification"
TEST_FILE = "dev_rehydrated.jsonl"
SUBMISSION_FILE = "submission.jsonl"
MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {0: "No", 1: "Yes"}
BATCH_SIZE = 64


def find_latest_checkpoint(base_path):
    """
    Scans the base_path directory (e.g., 'distilbert-conspiracy-classification')
    for the latest numbered 'checkpoint-*' subfolder where the model weights are stored.
    """
    checkpoint_dirs = glob.glob(os.path.join(base_path, "checkpoint-*"))

    if not checkpoint_dirs:
        # If no checkpoint folders are found, assume the model files are directly in the base path
        print(f"Warning: No 'checkpoint-*' folder found. Assuming model files are in: {base_path}")
        return base_path

    # Sort directories based on the checkpoint number (the integer part after the dash)
    # This reliably finds the checkpoint with the highest number, which is usually the final one.
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))

    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_competition_test_data(file_path):
    """
    Loads all data from a JSONL file for inference, preserving order,
    and retaining the document's unique ID.
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                sample_id = item.get("_id", f"sample_{i}")
                data.append({
                    "unique_sample_id": sample_id,
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data using the same approach as the training script."""
    # This uses tokenizer(examples["text"], truncation=True), mirroring the training script.
    # DataCollatorWithPadding handles the padding to max length in the batch.
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)


if __name__ == '__main__':

    # 1. Load Data
    raw_data = load_competition_test_data(TEST_FILE)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_list(raw_data)

    # Store the unique IDs for later submission file generation
    unique_ids = test_dataset["unique_sample_id"]

    # 2. Find and Load Tokenizer/Model
    model_directory = find_latest_checkpoint(MODEL_PATH)

    print(f"Loading tokenizer from {MODEL_NAME} and trained model from {model_directory}...")
    try:
        # Load the tokenizer.
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        # Load the model structure and weights from the discovered checkpoint directory.
        model = DistilBertForSequenceClassification.from_pretrained(model_directory)
    except Exception as e:
        print(f"Error loading model or tokenizer using path: '{model_directory}'.")
        print("Please verify that the directory contains 'config.json' and 'model.safetensors' or 'pytorch_model.bin'.")
        print(f"Details: {e}")
        sys.exit(-1)

    # 3. Tokenize Data
    tokenized_test_dataset = tokenize_data(test_dataset, tokenizer)

    # Remove columns that the model doesn't expect ('unique_sample_id' and 'text')
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["unique_sample_id", "text"])

    # 4. Prepare for Inference using Trainer with explicit data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    prediction_args = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=BATCH_SIZE,
            report_to="none"
        ),
        data_collator=data_collator  # Use the padding collator
    )

    # 5. Perform Inference
    print("Starting prediction...")
    predictions_output = prediction_args.predict(tokenized_test_dataset)

    # Get the class with the highest probability
    logits = predictions_output.predictions
    predicted_class_ids = np.argmax(logits, axis=-1)

    # 6. Map IDs to Labels
    predicted_labels = [LABEL_MAP[int(id)] for id in predicted_class_ids]

    # 7. Save Results in Codalab-ready JSONL format
    print(f"Saving {len(predicted_labels)} predictions to {SUBMISSION_FILE} (JSONL format)...")

    jsonl_lines = []
    for i, label in enumerate(predicted_labels):
        # Create a dictionary containing the ID and the prediction, using '_id' as the key
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": label
        }
        # Convert the dictionary to a JSON string and append to the list
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(SUBMISSION_FILE, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{SUBMISSION_FILE}' generated successfully.")
