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
    "F1-Score": [
        0.96,
        0.12,
        0.03,
        0.55,
        0.01
    ],
    "ROC-AUC": [
        0.9839,
        0.6412,
        0.4160,
        0.6812,
        0.4578
    ]
}

df = pd.DataFrame(results)

print("\n================ REAL MIMII MODEL COMPARISON ================\n")
print(df.to_string(index=False))

df.to_csv("real_mimii_comparison.csv", index=False)

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["F1-Score"]
)

plt.xlabel("Model")
plt.ylabel("Anomalous F1-Score")
plt.title("F1-Score Comparison on MIMII Fan ID_00")

plt.ylim(0, 1.05)

plt.xticks(
    rotation=20,
    ha="right"
)

for i, value in enumerate(df["F1-Score"]):
    plt.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "real_mimii_f1_comparison.png",
    dpi=300
)

plt.show()

plt.close()

print("\nSaved:")
print("real_mimii_comparison.csv")
print("real_mimii_f1_comparison.png")