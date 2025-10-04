import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np


def load_and_filter_data(file_path):
    """Loads data from a JSON file and filters out entries with 'Can't tell'."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'conspiracy' in item and item['conspiracy'] in ["Yes", "No"]:
                    data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data."""
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)


def encode_labels(dataset, label_to_id):
    """Encodes the labels to numerical values."""
    return dataset.map(lambda examples: {'labels': [label_to_id[label] for label in examples["conspiracy"]]},
                       batched=True)


def save_predictions(trainer, test_dataset, output_file):
    """Saves predictions and true labels to a JSON file."""
    predictions = trainer.predict(test_dataset)
    predicted_classes = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_dataset["labels"]
    results = []
    for i in range(len(true_labels)):
        results.append({
            "predicted_label": trainer.model.config.id2label[predicted_classes[i]],
            "true_label": trainer.model.config.id2label[true_labels[i]],
            "text": test_dataset["text"][i]
        })
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    train_file = "train_rehydrated.jsonl"
    model_name = "distilbert-base-uncased"
    output_dir = "distilbert-conspiracy-classification"
    label_to_id = {"No": 0, "Yes": 1}
    id_to_label = {0: "No", 1: "Yes"}
    num_labels = len(label_to_id)
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10

    # Load and filter data
    train_data = load_and_filter_data(train_file)

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_list(train_data)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Tokenize and encode labels
    tokenized_train_dataset = tokenize_data(train_dataset, tokenizer)
    encoded_train_dataset = encode_labels(tokenized_train_dataset, label_to_id)

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id_to_label,
                                                                label2id=label_to_id)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    print("Training the model...")
    trainer.train()
    print("Training finished.")
