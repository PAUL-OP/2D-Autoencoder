import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from mimii_pipeline import (
    find_mimii_files,
    wav_to_fixed_spec
)

DATA_ROOT = "./synthetic_data"
MACHINE = "fan"

EPOCHS = 30
BATCH_SIZE = 8
LR = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class LSTMAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            128,
            128
        )

    def forward(self, x):

        # x shape: (batch, 128, 128)
        encoded, (hidden, cell) = self.encoder(x)

        # Use the final encoded representation
        latent = encoded[:, -1, :]

        # Repeat latent vector for every time frame
        repeated = latent.unsqueeze(1).repeat(
            1,
            x.size(1),
            1
        )

        decoded, _ = self.decoder(
            repeated
        )

        output = self.output_layer(
            decoded
        )

        return torch.sigmoid(output)


def load_specs(files):

    specs = []

    for path in files:
        specs.append(
            wav_to_fixed_spec(path)
        )

    return np.array(
        specs,
        dtype=np.float32
    )


normal_files, abnormal_files = find_mimii_files(
    DATA_ROOT,
    MACHINE
)

rng = np.random.RandomState(42)
rng.shuffle(normal_files)

n_val = max(
    1,
    int(len(normal_files) * 0.15)
)

val_files = normal_files[:n_val]
train_files = normal_files[n_val:]


print("Loading normal training data...")
train_specs = load_specs(train_files)

print("Loading validation data...")
val_specs = load_specs(val_files)

print("Loading abnormal data...")
abnormal_specs = load_specs(
    abnormal_files
)


train_tensor = torch.tensor(
    train_specs,
    dtype=torch.float32
)

train_loader = torch.utils.data.DataLoader(
    train_tensor,
    batch_size=BATCH_SIZE,
    shuffle=True
)


model = LSTMAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)


print("\nTraining LSTM Autoencoder...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed = model(batch)

        loss = criterion(
            reconstructed,
            batch
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * batch.size(0)
        )

    epoch_loss = (
        total_loss /
        len(train_loader.dataset)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"- Loss: {epoch_loss:.6f}"
    )


model.eval()


def calculate_errors(specs):

    errors = []

    with torch.no_grad():

        for spec in specs:

            x = torch.tensor(
                spec,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            reconstructed = model(x)

            error = torch.mean(
                (reconstructed - x) ** 2
            ).item()

            errors.append(error)

    return np.array(errors)


normal_errors = calculate_errors(
    val_specs
)

abnormal_errors = calculate_errors(
    abnormal_specs
)


mu = np.mean(normal_errors)
sigma = np.std(normal_errors)

threshold = mu + 3 * sigma


test_errors = np.concatenate(
    [
        normal_errors,
        abnormal_errors
    ]
)

true_labels = np.array(
    [0] * len(normal_errors) +
    [1] * len(abnormal_errors)
)

predictions = (
    test_errors > threshold
).astype(int)


print("\n================ LSTM Autoencoder ================")

print(
    f"Threshold (mu + 3*sigma): "
    f"{threshold:.6f}"
)

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "Normal (0)",
            "Anomalous (1)"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        true_labels,
        predictions
    )
)

try:

    auc = roc_auc_score(
        true_labels,
        test_errors
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

except ValueError:
    pass


torch.save(
    model.state_dict(),
    "lstm_autoencoder.pt"
)

print(
    "\nModel saved to lstm_autoencoder.pt"
)