import pandas as pd
import matplotlib.pyplot as plt

results = {
    "Model": [
        "Isolation Forest",
        "MLP Autoencoder",
        "1D-CNN Autoencoder",
        "LSTM Autoencoder",
        "2D-CAE"
    ],
    "Precision": [
        0.99,
        1.00,
        1.00,
        1.00,
        0.23
    ],
    "Recall": [
        0.99,
        1.00,
        1.00,
        0.04,
        0.00
    ],
    "F1-Score": [
        0.99,
        1.00,
        1.00,
        0.08,
        0.00
    ],
    "ROC-AUC": [
        0.98,
        1.00,
        1.00,
        0.312,
        0.1973
    ]
}

df = pd.DataFrame(results)

print("\n================ MODEL COMPARISON ================\n")
print(df.to_string(index=False))

df.to_csv("model_comparison.csv", index=False)

metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]

for metric in metrics:
    plt.figure(figsize=(9, 5))
    plt.bar(df["Model"], df[metric])
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(metric + " Comparison")
    plt.xticks(rotation=20)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(metric.lower().replace("-", "_") + "_comparison.png")
    plt.close()

print("\nComparison table saved to model_comparison.csv")
print("Graphs saved successfully.")