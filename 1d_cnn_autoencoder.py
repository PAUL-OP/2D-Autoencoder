import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from mimii_pipeline import find_mimii_files

DATA_ROOT = "./mimii_data"
MACHINE = "fan"
MACHINE_ID = "id_00"

SR = 16000
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR)

    if len(audio) < SR:
        audio = np.pad(audio, (0, SR - len(audio)))

    if len(audio) > SR:
        audio = audio[:SR]

    return audio.astype(np.float32)


def load_files(files):
    data = []

    for i, path in enumerate(files):
        data.append(load_audio(path))

        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"Processed {i + 1}/{len(files)}")

    return np.array(data, dtype=np.float32)


class CNN1DAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32,
                kernel_size=9,
                stride=2,
                padding=4,
                output_padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 16,
                kernel_size=9,
                stride=2,
                padding=4,
                output_padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16, 1,
                kernel_size=9,
                stride=2,
                padding=4,
                output_padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

n_val = max(1, int(len(normal_files) * 0.15))

val_files = normal_files[:n_val]
train_files = normal_files[n_val:]

print("\nLoading training audio...")
train_audio = load_files(train_files)

print("\nLoading validation audio...")
val_audio = load_files(val_files)

print("\nLoading abnormal audio...")
abnormal_audio = load_files(abnormal_files)

train_tensor = torch.tensor(
    train_audio,
    dtype=torch.float32
).unsqueeze(1)

val_tensor = torch.tensor(
    val_audio,
    dtype=torch.float32
).unsqueeze(1)

abnormal_tensor = torch.tensor(
    abnormal_audio,
    dtype=torch.float32
).unsqueeze(1)

train_loader = torch.utils.data.DataLoader(
    train_tensor,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = CNN1DAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n================ Training 1D-CNN Autoencoder ================")

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

    average_loss = total_loss / len(train_loader)

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


print("\nCalculating validation reconstruction errors...")

val_errors = reconstruction_errors(val_tensor)

mu_loss = np.mean(val_errors)
sigma_loss = np.std(val_errors)

threshold = mu_loss + 3 * sigma_loss

print(
    f"\nCalculated Anomaly Threshold "
    f"(mu + 3*sigma): {threshold:.6f}"
)

print("\nCalculating test reconstruction errors...")

abnormal_errors = reconstruction_errors(
    abnormal_tensor
)

test_errors = np.concatenate(
    [val_errors, abnormal_errors]
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
    "1d_cnn_autoencoder.pt"
)

print(
    "\nModel saved to 1d_cnn_autoencoder.pt"
)