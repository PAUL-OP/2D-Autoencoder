import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error

from mimii_pipeline import (
    ConvAutoencoder2D,
    specs_from_files,
    find_mimii_files
)

DATA_ROOT = "./synthetic_data"
MACHINE = "fan"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normal_files, abnormal_files = find_mimii_files(
    DATA_ROOT,
    MACHINE
)

rng = np.random.RandomState(42)
rng.shuffle(normal_files)

n_val = max(1, int(len(normal_files) * 0.15))

val_files = normal_files[:n_val]
train_files = normal_files[n_val:]

val_specs = specs_from_files(val_files, desc="validation")
abnormal_specs = specs_from_files(abnormal_files, desc="abnormal")

model = ConvAutoencoder2D().to(device)

model.load_state_dict(
    torch.load(
        f"autoencoder_{MACHINE}.pt",
        map_location=device
    )
)

model.eval()


def calculate_errors(specs):

    errors = []

    with torch.no_grad():

        for spec in specs:

            x = torch.tensor(
                spec,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            reconstructed = model(x)

            error = torch.mean(
                (reconstructed - x) ** 2
            ).item()

            errors.append(error)

    return np.array(errors)


normal_errors = calculate_errors(val_specs)
abnormal_errors = calculate_errors(abnormal_specs)

threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)

print("Normal samples:", len(normal_errors))
print("Abnormal samples:", len(abnormal_errors))
print("Threshold:", threshold)

plt.figure(figsize=(10, 6))

plt.scatter(
    range(len(normal_errors)),
    normal_errors,
    label="Normal"
)

plt.scatter(
    range(
        len(normal_errors),
        len(normal_errors) + len(abnormal_errors)
    ),
    abnormal_errors,
    label="Abnormal"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold (μ + 3σ)"
)

plt.xlabel("Sample")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error: Normal vs Abnormal")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reconstruction_error.png",
    dpi=300
)

plt.show()

print("Graph saved as reconstruction_error.png")