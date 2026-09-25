import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from mimii_pipeline import find_mimii_files

DATA_ROOT = "./synthetic_data"
MACHINE = "fan"

SR = 16000
AUDIO_LENGTH = 16000

EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class CNN1DAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
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
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 16,
                kernel_size=9,
                stride=2,
                padding=4,
                output_padding=1
            ),
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

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SR
    )

    if len(audio) > AUDIO_LENGTH:

        audio = audio[:AUDIO_LENGTH]

    elif len(audio) < AUDIO_LENGTH:

        audio = np.pad(
            audio,
            (0, AUDIO_LENGTH - len(audio))
        )

    return audio.astype(np.float32)


def load_dataset(files):

    data = []

    for path in files:
        data.append(
            load_audio(path)
        )

    return np.array(data, dtype=np.float32)


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


print("Loading normal training audio...")
train_audio = load_dataset(train_files)

print("Loading validation audio...")
val_audio = load_dataset(val_files)

print("Loading abnormal audio...")
abnormal_audio = load_dataset(abnormal_files)


train_tensor = torch.tensor(
    train_audio,
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
    lr=LR
)


print("\nTraining 1D-CNN Autoencoder...")

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


def calculate_errors(audio_data):

    errors = []

    with torch.no_grad():

        for audio in audio_data:

            x = torch.tensor(
                audio,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            reconstructed = model(x)

            error = torch.mean(
                (reconstructed - x) ** 2
            ).item()

            errors.append(error)

    return np.array(errors)


normal_errors = calculate_errors(
    val_audio
)

abnormal_errors = calculate_errors(
    abnormal_audio
)


mu = np.mean(normal_errors)
sigma = np.std(normal_errors)

threshold = mu + 3 * sigma


test_errors = np.concatenate(
    [normal_errors, abnormal_errors]
)

true_labels = np.array(
    [0] * len(normal_errors) +
    [1] * len(abnormal_errors)
)

predictions = (
    test_errors > threshold
).astype(int)


print("\n================ 1D-CNN Autoencoder ================")

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
    "1d_cnn_autoencoder.pt"
)

print(
    "\nModel saved to 1d_cnn_autoencoder.pt"
)