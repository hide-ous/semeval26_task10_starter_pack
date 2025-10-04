import json
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
# from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
# from seqeval.scheme import IOB2

# from baseline_binary import save_predictions


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data

def create_label_maps_simplified(marker_type):
    label_list = ["O", marker_type]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_list)

def tokenize_and_align_labels_simplified(examples, tokenizer, label_to_id, marker_type):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_offsets_mapping=True)
    labels = []
    all_markers = examples.get("markers", [])

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        example_labels = [0] * len(offsets) # Initialize with 'O'
        example_markers = all_markers[i] if i < len(all_markers) else []

        for marker in example_markers:
            if marker["type"] == marker_type:
                start_char = marker["startIndex"]
                end_char = marker["endIndex"]
                marker_label = label_to_id.get(marker_type)
                if marker_label is not None:
                    for token_idx, (start, end) in enumerate(offsets):
                        if start is not None and end is not None:
                            if start_char <= start < end_char:
                                if token_idx < len(example_labels):
                                    example_labels[token_idx] = marker_label
                            elif start < end_char and end > start_char: # Handle partial overlaps
                                if token_idx < len(example_labels) and example_labels[token_idx] == 0:
                                    example_labels[token_idx] = marker_label
        labels.append(example_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# def compute_metrics_simplified(p, id_to_label, marker_type):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#
#     true_labels_iob2 = []
#     predicted_labels_iob2 = []
#
#     for label_seq, pred_seq in zip(labels, predictions):
#         true_iob2_seq = []
#         pred_iob2_seq = []
#         for t_label_id, p_label_id in zip(label_seq, pred_seq):
#             if t_label_id != -100:
#                 true_label = id_to_label[t_label_id]
#                 pred_label = id_to_label[p_label_id]
#
#                 if true_label == "O":
#                     true_iob2_seq.append("O")
#                 else:
#                     true_iob2_seq.append(f"B-{marker_type}") # Treat all as beginning for reporting
#
#                 if pred_label == "O":
#                     pred_iob2_seq.append("O")
#                 else:
#                     pred_iob2_seq.append(f"B-{marker_type}") # Treat all as beginning for reporting
#         true_labels_iob2.append(true_iob2_seq)
#         predicted_labels_iob2.append(pred_iob2_seq)
#
#     return {
#         "precision": precision_score(true_labels_iob2, predicted_labels_iob2, scheme=IOB2, zero_division=0),
#         "recall": recall_score(true_labels_iob2, predicted_labels_iob2, scheme=IOB2, zero_division=0),
#         "f1": f1_score(true_labels_iob2, predicted_labels_iob2, scheme=IOB2, zero_division=0),
#         "report": classification_report(true_labels_iob2, predicted_labels_iob2, scheme=IOB2, zero_division=0),
#     }

if __name__ == "__main__":
    train_file = "train_rehydrated.jsonl"
    # test_file = "test.jsonl"
    model_name = "distilbert-base-uncased"
    output_dir_base = "distilbert-single-type-simplified"
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10
    marker_types_to_train = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    all_results = {}

    # Load data once
    train_data = load_data(train_file)
    # test_data = load_data(test_file)

    train_dataset = Dataset.from_list(train_data)
    # test_dataset = Dataset.from_list(test_data)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    for marker_type in marker_types_to_train:
        print(f"\n--- Training simplified model for marker type: {marker_type} ---")

        # Create simplified label maps
        label_to_id, id_to_label, num_labels = create_label_maps_simplified(marker_type)
        print("Label to ID mapping:", label_to_id)
        print("ID to Label mapping:", id_to_label)
        print("Number of labels:", num_labels)

        # Tokenize and align labels (simplified)
        tokenized_train_dataset = train_dataset.map(
            tokenize_and_align_labels_simplified,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id, "marker_type": marker_type}
        )
        # tokenized_test_dataset = test_dataset.map(
        #     tokenize_and_align_labels_simplified,
        #     batched=True,
        #     fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id, "marker_type": marker_type}
        # )

        # Load a new model for each marker type (now with 2 output labels)
        model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

        # Define training arguments
        output_dir = f"{output_dir_base}-{marker_type}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            # per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            # eval_strategy="epoch",
            logging_dir=f'./logs-{marker_type}-simplified',
            report_to="none"
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        print(f"Training simplified model for {marker_type}...")
        trainer.train()
        print(f"Training for {marker_type} finished.")

