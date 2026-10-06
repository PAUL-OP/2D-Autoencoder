import os
import glob
import random

import numpy as np
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "./mimii_data"

MACHINE = "fan"
MACHINE_ID = "id_00"

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

WINDOW_HOP = WINDOW_SAMPLES // 2

SEED = 42

VAL_FRACTION = 0.15

BATCH_SIZE = 64

EPOCHS = 80

LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FIND FILES
# ============================================================

def find_files():

    base = os.path.join(
        DATA_ROOT,
        MACHINE,
        MACHINE_ID
    )

    normal_files = sorted(
        glob.glob(
            os.path.join(
                base,
                "normal",
                "*.wav"
            )
        )
    )

    abnormal_files = sorted(
        glob.glob(
            os.path.join(
                base,
                "abnormal",
                "*.wav"
            )
        )
    )

    return normal_files, abnormal_files


# ============================================================
# AUDIO -> MEL SPECTROGRAM
# ============================================================

def audio_to_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = np.clip(
        mel_db,
        -80.0,
        0.0
    )

    mel_norm = (
        mel_db + 80.0
    ) / 80.0

    return mel_norm.astype(
        np.float32
    )


# ============================================================
# 1-SECOND OVERLAPPING WINDOWS
# ============================================================

def audio_to_windows(audio):

    if len(audio) < WINDOW_SAMPLES:

        padded = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.float32
        )

        padded[:len(audio)] = audio

        return [
            audio_to_mel(padded)
        ]

    windows = []

    for start in range(
        0,
        len(audio) - WINDOW_SAMPLES + 1,
        WINDOW_HOP
    ):

        segment = audio[
            start:
            start + WINDOW_SAMPLES
        ]

        windows.append(
            audio_to_mel(segment)
        )

    last_start = (
        len(audio) - WINDOW_SAMPLES
    )

    if len(windows) == 0 or (
        last_start -
        (
            (len(windows) - 1)
            * WINDOW_HOP
        )
        > 0
    ):

        segment = audio[
            last_start:
            last_start + WINDOW_SAMPLES
        ]

        windows.append(
            audio_to_mel(segment)
        )

    return windows


# ============================================================
# RECORDING -> MEL WINDOWS
# ============================================================

def recording_to_windows(path):

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    windows = audio_to_windows(
        audio
    )

    return np.stack(
        windows
    )


# ============================================================
# MLP DATASET
#
# Paper:
# Mel spectrogram is flattened into a 1D vector.
# ============================================================

class MLPDataset(Dataset):

    def __init__(self, files):

        self.samples = []

        print(
            f"Preparing dataset from "
            f"{len(files)} recordings..."
        )

        for index, path in enumerate(
            files
        ):

            windows = recording_to_windows(
                path
            )

            for window in windows:

                # Flatten 2D Mel spectrogram
                # into a 1D vector.
                flattened = window.flatten()

                self.samples.append(
                    flattened
                )

            if (
                (index + 1) % 100 == 0
                or index + 1 == len(files)
            ):

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(files)}"
                )

        print(
            f"Total training vectors: "
            f"{len(self.samples)}"
        )

        print(
            f"Input dimension: "
            f"{len(self.samples[0])}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return torch.tensor(
            self.samples[index],
            dtype=torch.float32
        )


# ============================================================
# MLP AUTOENCODER
# ============================================================

class MLPAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                64
            )
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                64,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                input_dim
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(
            encoded
        )

        return reconstructed


# ============================================================
# RECORDING ERROR
# ============================================================

def recording_error(
    model,
    path
):

    model.eval()

    windows = recording_to_windows(
        path
    )

    errors = []

    with torch.no_grad():

        for window in windows:

            x = torch.tensor(
                window.flatten(),
                dtype=torch.float32,
                device=DEVICE
            )

            x = x.unsqueeze(0)

            reconstruction = model(
                x
            )

            error = torch.mean(
                (
                    reconstruction - x
                ) ** 2
            ).item()

            errors.append(
                error
            )

    return float(
        np.mean(errors)
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    validation_files
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
    )

    best_val = float("inf")

    best_state = None

    patience = 12

    no_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = criterion(
                reconstruction,
                batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch.size(0)
            )

        train_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_errors = []

        for path in validation_files:

            error = recording_error(
                model,
                path
            )

            validation_errors.append(
                error
            )

        validation_loss = float(
            np.mean(
                validation_errors
            )
        )

        scheduler.step(
            validation_loss
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.8f} | "
            f"Val MSE: "
            f"{validation_loss:.8f}"
        )

        if validation_loss < best_val:

            best_val = validation_loss

            best_state = {
                key:
                value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            no_improvement = 0

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print(
                "Early stopping."
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("STANDARD MLP AUTOENCODER")
    print("MIMII FAN ID_00")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # FIND DATA
    # ========================================================

    normal_files, abnormal_files = find_files()

    print()

    print(
        f"Normal recordings: "
        f"{len(normal_files)}"
    )

    print(
        f"Abnormal recordings: "
        f"{len(abnormal_files)}"
    )

    # ========================================================
    # SAME RECORDING-LEVEL SPLIT
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    shuffled_normal = list(
        normal_files
    )

    rng.shuffle(
        shuffled_normal
    )

    number_normal_test = min(
        len(abnormal_files),
        len(shuffled_normal) // 2
    )

    normal_test_files = (
        shuffled_normal[
            :number_normal_test
        ]
    )

    remaining_normal = (
        shuffled_normal[
            number_normal_test:
        ]
    )

    number_validation = max(
        1,
        int(
            len(remaining_normal)
            * VAL_FRACTION
        )
    )

    validation_files = (
        remaining_normal[
            :number_validation
        ]
    )

    train_files = (
        remaining_normal[
            number_validation:
        ]
    )

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training normal: "
        f"{len(train_files)}"
    )

    print(
        f"Validation normal: "
        f"{len(validation_files)}"
    )

    print(
        f"Normal test: "
        f"{len(normal_test_files)}"
    )

    print(
        f"Abnormal test: "
        f"{len(abnormal_files)}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING MLP TRAINING DATASET")
    print("=" * 70)

    train_dataset = MLPDataset(
        train_files
    )

    input_dim = len(
        train_dataset.samples[0]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING MLP AUTOENCODER")
    print("=" * 70)

    model = MLPAutoencoder(
        input_dim
    ).to(
        DEVICE
    )

    print(model)

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    model = train_model(
        model,
        train_loader,
        validation_files
    )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD")
    print("=" * 70)

    validation_errors = []

    for path in validation_files:

        validation_errors.append(
            recording_error(
                model,
                path
            )
        )

    validation_errors = np.array(
        validation_errors
    )

    mean_error = np.mean(
        validation_errors
    )

    std_error = np.std(
        validation_errors
    )

    threshold = (
        mean_error
        + 3.0 * std_error
    )

    print(
        f"Mean: "
        f"{mean_error:.8f}"
    )

    print(
        f"Std: "
        f"{std_error:.8f}"
    )

    print(
        f"Threshold: "
        f"{threshold:.8f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 70)
    print("TESTING")
    print("=" * 70)

    test_files = (
        normal_test_files
        + abnormal_files
    )

    y_true = np.array(
        [0] * len(normal_test_files)
        + [1] * len(abnormal_files)
    )

    test_errors = []

    for i, path in enumerate(
        test_files
    ):

        test_errors.append(
            recording_error(
                model,
                path
            )
        )

        if (
            (i + 1) % 50 == 0
            or i + 1 == len(test_files)
        ):

            print(
                f"Testing "
                f"{i + 1}/"
                f"{len(test_files)}"
            )

    test_errors = np.array(
        test_errors
    )

    y_pred = (
        test_errors > threshold
    ).astype(int)

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomalous"
            ],
            digits=4
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    try:

        auc = roc_auc_score(
            y_true,
            test_errors
        )

        print()

        print(
            f"ROC-AUC: "
            f"{auc:.4f}"
        )

    except Exception:

        auc = None

    # ========================================================
    # SAVE
    # ========================================================

    model_path = (
        "mlp_autoencoder_paper_id_00.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                input_dim
        },
        model_path
    )

    print()

    print(
        f"Model saved to: "
        f"{model_path}"
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()