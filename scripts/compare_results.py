import os
import json
import pandas as pd

# look in current folder for all experiment directories
base_dir = "."

# pick folders like: bert-sentiment-output_lr2e-5_bs4_ep3_ml256
experiment_dirs = [
    d for d in os.listdir(base_dir)
    if d.startswith("bert-sentiment-output")
    and os.path.isdir(os.path.join(base_dir, d))
]

results = []

for d in sorted(experiment_dirs):
    train_metrics_path = os.path.join(base_dir, d, "train_metrics.json")
    if not os.path.exists(train_metrics_path):
        continue

    with open(train_metrics_path, "r") as f:
        metrics = json.load(f)

    results.append({
        "experiment": d,
        "train_loss": metrics.get("train_loss"),
        "train_accuracy": metrics.get("train_accuracy"),
        "train_f1": metrics.get("train_f1"),
        "train_runtime_s": metrics.get("train_runtime"),
        "epoch": metrics.get("epoch"),
    })

# make a table
df = pd.DataFrame(results)

# sort best-first by accuracy then f1 (if they exist)
if not df.empty:
    df = df.sort_values(by=["train_accuracy", "train_f1"], ascending=False)

    print("\n📊 Summary of BERT experiment results:")
    print(df.to_string(index=False))

    # also save to csv so you can push to GitHub
    out_path = "bert_experiment_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved summary to {out_path}")
else:
    print("No experiment results found. Make sure your folders contain train_metrics.json.")
