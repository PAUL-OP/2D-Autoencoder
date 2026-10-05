import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from mimii_pipeline import find_mimii_files, wav_to_fixed_spec


DATA_ROOT = "./mimii_data"
MACHINE = "fan"
MACHINE_ID = "id_00"

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


normal_files, abnormal_files = find_mimii_files(
    DATA_ROOT,
    MACHINE,
    MACHINE_ID
)

print(
    f"Found {len(normal_files)} normal / "
    f"{len(abnormal_files)} abnormal recordings."
)


rng = np.random.RandomState(42)
rng.shuffle(normal_files)

n_val = max(
    1,
    int(len(normal_files) * 0.15)
)

val_files = normal_files[:n_val]
train_files = normal_files[n_val:]


def load_specs(files, name):

    specs = []

    for i, path in enumerate(files):

        spec = wav_to_fixed_spec(path)

        specs.append(spec)

        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(
                f"[{name}] processed "
                f"{i + 1}/{len(files)}"
            )

    return np.array(
        specs,
        dtype=np.float32
    )


print("\nExtracting Mel-Spectrograms...")

train_specs = load_specs(
    train_files,
    "train"
)

val_specs = load_specs(
    val_files,
    "validation"
)

abnormal_specs = load_specs(
    abnormal_files,
    "abnormal"
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

        self.output_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded, _ = self.encoder(x)

        decoded, _ = self.decoder(encoded)

        output = self.output_layer(decoded)

        return output


train_tensor = torch.tensor(
    train_specs,
    dtype=torch.float32
)

val_tensor = torch.tensor(
    val_specs,
    dtype=torch.float32
)

abnormal_tensor = torch.tensor(
    abnormal_specs,
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
    lr=LEARNING_RATE
)


print(
    "\n================ Training LSTM Autoencoder ================"
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

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

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


model.eval()


def reconstruction_errors(data):

    errors = []

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            reconstructed = model(batch)

            batch_errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=(1, 2)
            )

            errors.extend(
                batch_errors.cpu().numpy()
            )

    return np.array(errors)


print(
    "\nCalculating validation reconstruction errors..."
)

val_errors = reconstruction_errors(
    val_tensor
)


mu_loss = np.mean(val_errors)

sigma_loss = np.std(val_errors)

threshold = (
    mu_loss +
    3 * sigma_loss
)


print(
    f"\nCalculated Anomaly Threshold "
    f"(mu + 3*sigma): {threshold:.6f}"
)


print(
    "\nCalculating abnormal reconstruction errors..."
)

abnormal_errors = reconstruction_errors(
    abnormal_tensor
)


test_errors = np.concatenate(
    [
        val_errors,
        abnormal_errors
    ]
)


test_labels = np.array(
    [0] * len(val_errors) +
    [1] * len(abnormal_errors)
)


predictions = (
    test_errors > threshold
).astype(int)


print(
    "\n================ Classification Report ================"
)

print(
    classification_report(
        test_labels,
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
        test_labels,
        predictions
    )
)


try:

    auc = roc_auc_score(
        test_labels,
        test_errors
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

except ValueError:

    auc = None


torch.save(
    model.state_dict(),
    "lstm_autoencoder.pt"
)


print(
    "\nModel saved to lstm_autoencoder.pt"
)