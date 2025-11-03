#!/usr/bin/env bash
set -euo pipefail
BRANCH="${1:-main}"
FILE="train_bert_sentiment.py"
if [ ! -f "$FILE" ]; then
  echo "Error: $FILE not found in repository root." >&2
  exit 1
fi

git add "$FILE"
if git diff --cached --quiet; then
  echo "No changes detected in $FILE."
  exit 0
fi

COMMIT_MESSAGE=${2:-"Update train_bert_sentiment.py"}

git commit -m "$COMMIT_MESSAGE"

git push origin "$BRANCH"
