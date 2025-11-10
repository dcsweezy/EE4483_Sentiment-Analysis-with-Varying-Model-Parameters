#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def load_labeled_json(path: str) -> Tuple[List[str], List[int]]:
    """Load a JSON list with fields 'reviews' and 'sentiments'."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["reviews"] for r in records]
    labels = [int(r["sentiments"]) for r in records]
    return texts, labels


def load_unlabeled_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["reviews"] for r in records]
    return texts


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # use macro because your data is imbalanced (85% label 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class WeightedTrainer(Trainer):
    """
    Trainer that applies class-weighted cross entropy.
    """

    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("K-fold + final train + test prediction")
    parser.add_argument("--train", type=str, default="Dataset/train.json", help="path to labelled train.json")
    parser.add_argument("--test", type=str, default="Dataset/test.json", help="path to unlabeled test.json")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="outputs_full_pipeline")
    parser.add_argument("--k", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. load labeled data (for k-fold and for final train)
    # --------------------------------------------------------
    train_texts, train_labels = load_labeled_json(args.train)
    print(f"Loaded {len(train_texts)} labelled training samples.")

    # tokenizer & collator shared
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # compute class weights from full train (reuse for folds & final)
    counts = {}
    for lbl in train_labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    num_classes = 2
    total = len(train_labels)
    class_weights = []
    for cls in range(num_classes):
        c = counts.get(cls, 1)
        # inverse frequency style
        w = total / (num_classes * c)
        class_weights.append(w)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights (from full train): {class_weights.tolist()}")

    # --------------------------------------------------------
    # 2. K-FOLD CROSS VALIDATION
    # --------------------------------------------------------
    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_texts, train_labels), start=1):
        print(f"\n========== FOLD {fold_idx}/{args.k} ==========")

        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        # build datasets
        ds_train = Dataset.from_dict({"text": fold_train_texts, "labels": fold_train_labels})
        ds_val = Dataset.from_dict({"text": fold_val_texts, "labels": fold_val_labels})

        def tok(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

        ds_train_tok = ds_train.map(tok, batched=True)
        ds_val_tok = ds_val.map(tok, batched=True)

        # rename to "labels" -> "labels" is okay, HF will map to "labels"
        ds_train_tok = ds_train_tok.remove_columns(["text"])
        ds_val_tok = ds_val_tok.remove_columns(["text"])

        # fresh model per fold
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(fold_output_dir, "logs"),
            logging_steps=50,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=ds_train_tok,
            eval_dataset=ds_val_tok,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate(ds_val_tok)
        print(f"Fold {fold_idx} metrics: {eval_metrics}")

        fold_results.append(
            {
                "fold": fold_idx,
                "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
                "eval_precision": float(eval_metrics.get("eval_precision", 0.0)),
                "eval_recall": float(eval_metrics.get("eval_recall", 0.0)),
                "eval_f1": float(eval_metrics.get("eval_f1", 0.0)),
            }
        )

    # save k-fold summary
    kfold_df = pd.DataFrame(fold_results)
    kfold_path = os.path.join(args.output_dir, "kfold_results.csv")
    kfold_df.to_csv(kfold_path, index=False)
    print(f"\n✅ Saved k-fold metrics to {kfold_path}")
    print("Average F1 over folds:", kfold_df["eval_f1"].mean())

    # --------------------------------------------------------
    # 3. TRAIN FINAL MODEL ON FULL TRAIN SET (weighted)
    # --------------------------------------------------------
    print("\n========== TRAINING FINAL MODEL ON FULL TRAIN ==========")
    full_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})

    def tok_full(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    full_ds_tok = full_ds.map(tok_full, batched=True)
    full_ds_tok = full_ds_tok.remove_columns(["text"])

    final_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    final_output_dir = os.path.join(args.output_dir, "final_model")
    final_training_args = TrainingArguments(
        output_dir=final_output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(final_output_dir, "logs"),
        logging_steps=50,
    )

    final_trainer = WeightedTrainer(
        class_weights=class_weights,
        model=final_model,
        args=final_training_args,
        train_dataset=full_ds_tok,
        eval_dataset=full_ds_tok,  # just to log train metrics
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = final_trainer.train()
    final_eval = final_trainer.evaluate(full_ds_tok)

    # collect final train metrics
    train_metrics = train_result.metrics if train_result.metrics is not None else {}
    train_metrics.update(
        {
            "train_accuracy": float(final_eval.get("eval_accuracy", 0.0)),
            "train_precision": float(final_eval.get("eval_precision", 0.0)),
            "train_recall": float(final_eval.get("eval_recall", 0.0)),
            "train_f1": float(final_eval.get("eval_f1", 0.0)),
        }
    )

    metrics_dir = Path(final_output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"✅ Saved final train metrics to {metrics_dir / 'train_metrics.json'}")

    # --------------------------------------------------------
    # 4. PREDICT ON UNLABELLED TEST AND SAVE submission.csv
    # --------------------------------------------------------
    print("\n========== PREDICTING ON TEST ==========")
    test_texts = load_unlabeled_json(args.test)
    test_ds = Dataset.from_dict({"text": test_texts})
    test_ds_tok = test_ds.map(tok_full, batched=True)
    test_ds_tok = test_ds_tok.remove_columns(["text"])

    preds = final_trainer.predict(test_ds_tok)
    logits = preds.predictions
    predicted_labels = np.argmax(logits, axis=-1)

    # softmax confidence
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidence = probs[np.arange(len(predicted_labels)), predicted_labels]

    # reload original test file to get text back (for CSV)
    with open(args.test, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    df_sub = pd.DataFrame(
        {
            "review_id": range(len(test_records)),
            "review_text": [r["reviews"] for r in test_records],
            "predicted_sentiment": predicted_labels,
            "confidence": confidence,
        }
    )

    sub_path = os.path.join(args.output_dir, "submission.csv")
    df_sub.to_csv(sub_path, index=False, encoding="utf-8")
    print(f"✅ Saved predictions to {sub_path}")

    # save note about test
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "message": "Test set was unlabeled; submission.csv contains predictions.",
                "num_samples": len(test_records),
            },
            f,
            indent=2,
        )

    # save final model + tokenizer
    final_trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
