import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    parser.add_argument("--train", type=str, default="train.json", help="Path to the training JSON file")
    parser.add_argument("--test", type=str, default="test.json", help="Path to the testing JSON file")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Pretrained model checkpoint to use")
    parser.add_argument(
        "--output_dir", type=str, default="bert-sentiment-output", help="Directory to save the model and logs"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_json_dataset(path, has_labels=True):
    with open(path) as f:
        records = json.load(f)

    texts = [record["reviews"] for record in records]

    if has_labels:
        labels = [int(record["sentiments"]) for record in records]
        return texts, labels
    else:
        return texts

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) load train (with labels) and test (without)
    train_texts, train_labels = load_json_dataset(args.train, has_labels=True)
    test_texts = load_json_dataset(args.test, has_labels=False)

    # 2) build HF Datasets from raw lists
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts})

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    # 4) tokenize both
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # The Trainer API expects columns named "input_ids", "attention_mask", and "label".
    # 5) remove text col (Trainer wants input_ids, attention_mask, label)
    tokenized_train = tokenized_train.remove_columns([column for column in tokenized_train.column_names if column == "text"])
    
    # test has no label, so we only remove text there
    tokenized_test = tokenized_test.remove_columns([column for column in tokenized_test.column_names if column == "text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        #evaluation_strategy="epoch",  # keep this one
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",
        # greater_is_better=True,
    )

    # ❗ since test has no labels, use train as eval for now
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_train,  # <- was tokenized_test, but that had no labels
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- TRAINING ----
    train_result = trainer.train()
    train_metrics = train_result.metrics if train_result.metrics is not None else {}

    # Compute metrics on the training set for accuracy, precision, recall, F1
    train_eval = trainer.evaluate(tokenized_train)
    train_metrics.update({
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else float(train_metrics.get("train_loss", 0)),
        "train_accuracy": float(train_eval.get("eval_accuracy", 0)),
        "train_precision": float(train_eval.get("eval_precision", 0)),
        "train_recall": float(train_eval.get("eval_recall", 0)),
        "train_f1": float(train_eval.get("eval_f1", 0)),
        "train_runtime": float(train_metrics.get("train_runtime", 0)),
        "train_samples_per_second": float(train_metrics.get("train_samples_per_second", 0)),
        "train_steps_per_second": float(train_metrics.get("train_steps_per_second", 0)),
        "epoch": float(train_metrics.get("epoch", 0)),
    })

    print("\n📊 Training metrics:")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")


    # ---- TESTING ----
    eval_result = trainer.evaluate(tokenized_test)
    eval_metrics = {
        "eval_loss": float(eval_result.get("eval_loss", 0)),
        "eval_accuracy": float(eval_result.get("eval_accuracy", 0)),
        "eval_precision": float(eval_result.get("eval_precision", 0)),
        "eval_recall": float(eval_result.get("eval_recall", 0)),
        "eval_f1": float(eval_result.get("eval_f1", 0)),
        "eval_runtime": float(eval_result.get("eval_runtime", 0)),
        "eval_samples_per_second": float(eval_result.get("eval_samples_per_second", 0)),
        "eval_steps_per_second": float(eval_result.get("eval_steps_per_second", 0)),
        "epoch": float(eval_result.get("epoch", 0)),
    }

    print("\n📈 Test metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    ## ---- SAVE TO FILES ----
    metrics_dir = Path(args.output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


    # ---- Generate predictions on test.json and save to CSV ----
    print("\nGenerating predictions on test.json...")

    # run model on test set
    preds = trainer.predict(tokenized_test)
    logits = preds.predictions  # shape: (num_examples, 2)

    # turn logits into probabilities
    # softmax over axis=1
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # shape: (num_examples, 2)

    # predicted class = argmax
    predicted_labels = np.argmax(logits, axis=-1)

    # confidence = probability of the predicted class
    confidence = probs[np.arange(len(predicted_labels)), predicted_labels]

    # reload original test.json to get texts
    with open(args.test, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    assert len(test_records) == len(predicted_labels)

    df = pd.DataFrame({
        "review_id": range(len(test_records)),
        "review_text": [r["reviews"] for r in test_records],
        "predicted_sentiment": predicted_labels,
        "confidence": confidence,
    })

    output_csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"\n✅ Predictions with confidence saved to {output_csv_path}")

if __name__ == "__main__":
    main()
