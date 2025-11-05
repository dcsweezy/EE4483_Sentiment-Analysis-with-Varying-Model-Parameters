import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Output_results_Optimal_Hybrid/submission.csv")

# Summary
print("=== Prediction Summary ===")
print(df["predicted_sentiment"].value_counts(normalize=True))
print(f"\nAverage confidence: {df['confidence'].mean():.4f}")
print(f"Std dev confidence: {df['confidence'].std():.4f}")

# Confidence histogram
plt.hist(df["confidence"], bins=20, color="skyblue", edgecolor="black")
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()

# Optional: entropy measure
entropy = -np.mean(df["confidence"] * np.log(df["confidence"] + 1e-10))
print(f"\nAverage prediction entropy: {entropy:.4f}")

