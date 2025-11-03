# EE4483_Sentiment-Analysis-with-Varying-Model-Parameters

This project builds a BERT-based sentiment classifier to predict whether product reviews are positive or negative. Each member fine-tunes BERT with different preprocessing and parameters, then compares and merges results to create one optimized, high-accuracy model.

## Getting Started

1. Install the required Python packages (PyTorch, Transformers, Datasets, and scikit-learn). A minimal environment can be created with:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers datasets scikit-learn
   ```

2. Train and evaluate the model using the provided `train.json` and `test.json` files:

   ```bash
   python train_bert_sentiment.py \
       --train train.json \
       --test test.json \
       --model bert-base-uncased \
       --epochs 3 \
       --batch_size 8 \
       --learning_rate 2e-5
   ```

3. The fine-tuned model and tokenizer will be saved to `bert-sentiment-output/` by default, along with training logs and evaluation metrics printed to the console.

Use the command-line arguments to experiment with different pretrained checkpoints, hyper-parameters, and random seeds.

## Automatically committing the training script

To quickly save updates to `train_bert_sentiment.py` in your GitHub repository, run the helper script:

```bash
./scripts/save_train_script.sh <branch-name> "Commit message"
```

- `branch-name` (optional) defaults to `main`.
- The commit message (optional) defaults to `Update train_bert_sentiment.py`.

The script stages `train_bert_sentiment.py`, creates a commit if there are changes, and pushes it to the specified branch. Ensure that your local repository is authenticated with GitHub before running the script.
