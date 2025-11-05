#!/usr/bin/env bash
set -e  # stop script if any command fails

# ===============================
#  Auto-activate your Conda env
# ===============================
CONDA_ENV="EE4483_Sentiment"
echo "🔧 Activating Conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# ===============================
#  Define hyperparameter grid
# ===============================
LEARNING_RATES=("2e-5" "3e-5" "5e-5")
BATCH_SIZES=("4" "8")
EPOCHS=("3" "4")
MAX_LENGTHS=("256")

# ===============================
#  Loop through all combinations
# ===============================
for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      for ml in "${MAX_LENGTHS[@]}"; do
        OUTPUT_DIR="bert-sentiment-output_lr${lr}_bs${bs}_ep${ep}_ml${ml}"
        echo "🚀 Running BERT with lr=${lr}, bs=${bs}, epochs=${ep}, max_len=${ml}"
        echo "📂 Output folder: ${OUTPUT_DIR}"

        python train_bert_sentiment.py \
          --train train.json \
          --test test.json \
          --model bert-base-uncased \
          --epochs ${ep} \
          --batch_size ${bs} \
          --learning_rate ${lr} \
          --max_length ${ml} \
          --output_dir ${OUTPUT_DIR}

        echo "✅ Finished run for lr=${lr}, bs=${bs}, ep=${ep}"
        echo "---------------------------------------------"
      done
    done
  done
done

echo "🎯 All experiments completed successfully!"
