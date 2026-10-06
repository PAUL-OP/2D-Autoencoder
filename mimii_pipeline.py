import os
import glob
import random
import argparse
import json

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

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

# 50% overlap
WINDOW_HOP = WINDOW_SAMPLES // 2

SEED = 42

# Return to the successful 15% split
VAL_FRACTION = 0.15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FIND MIMII FILES
# ============================================================

def find_files(
    data_root,
    machine,
    machine_id=None
):

    base = os.path.join(
        data_root,
        machine
    )

    if machine_id is not None:

        base = os.path.join(
            base,
            machine_id
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


def find_mimii_files(
    data_root,
    machine,
    machine_id=None
):

    return find_files(
        data_root,
        machine,
        machine_id
    )


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

    # Fixed dynamic range
    mel_db = np.clip(
        mel_db,
        -80.0,
        0.0
    )

    # [-80, 0] -> [0, 1]
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

    # Include final region
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
# RECORDING -> WINDOWS
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
# DATASET
# ============================================================

class MelWindowDataset(Dataset):

    def __init__(self, files):

        self.files = files
        self.samples = []

        print(
            f"Preparing dataset from "
            f"{len(files)} recordings..."
        )

        for index, path in enumerate(files):

            windows = recording_to_windows(
                path
            )

            for window in windows:

                self.samples.append(
                    window
                )

            if (
                index + 1
            ) % 100 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(files)} recordings"
                )

        print(
            f"Total training windows: "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        x = torch.tensor(
            self.samples[index],
            dtype=torch.float32
        )

        # [128, time]
        # ->
        # [1, 128, time]

        return x.unsqueeze(0)


# ============================================================
# 2D CONVOLUTIONAL AUTOENCODER
# ============================================================

class ConvAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(
            encoded
        )

        return reconstructed


# ============================================================
# COMPOSITE RECONSTRUCTION LOSS
# ============================================================

def reconstruction_loss(
    reconstruction,
    target
):

    # --------------------------------------------------------
    # 1. Standard reconstruction error
    # --------------------------------------------------------

    mse_loss = torch.mean(
        (
            reconstruction - target
        ) ** 2
    )

    # --------------------------------------------------------
    # 2. Frequency-direction gradient
    #
    # Difference between neighbouring Mel-frequency bins.
    # --------------------------------------------------------

    target_freq = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    reconstruction_freq = (
        reconstruction[:, :, 1:, :]
        - reconstruction[:, :, :-1, :]
    )

    frequency_loss = torch.mean(
        torch.abs(
            reconstruction_freq
            - target_freq
        )
    )

    # --------------------------------------------------------
    # 3. Time-direction gradient
    #
    # Difference between neighbouring time frames.
    # --------------------------------------------------------

    target_time = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    reconstruction_time = (
        reconstruction[:, :, :, 1:]
        - reconstruction[:, :, :, :-1]
    )

    time_loss = torch.mean(
        torch.abs(
            reconstruction_time
            - target_time
        )
    )

    gradient_loss = (
        frequency_loss
        + time_loss
    )

    # --------------------------------------------------------
    # Composite objective
    #
    # MSE remains dominant.
    # Gradient component encourages preservation
    # of local time-frequency structure.
    # --------------------------------------------------------

    total_loss = (
        mse_loss
        + 0.20 * gradient_loss
    )

    return total_loss


# ============================================================
# RECORDING-LEVEL RECONSTRUCTION ERROR
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
                window,
                dtype=torch.float32,
                device=DEVICE
            )

            x = x.unsqueeze(0)
            x = x.unsqueeze(0)

            reconstruction = model(x)

            # IMPORTANT:
            # anomaly score remains standard MSE.
            #
            # We do NOT use the composite training loss
            # as the anomaly score.

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
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    val_files,
    epochs,
    learning_rate
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
    )

    best_val_loss = float("inf")

    best_state = None

    patience = 12

    epochs_without_improvement = 0

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            batch = batch.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = reconstruction_loss(
                reconstruction,
                batch
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * batch.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # VALIDATION
        #
        # Validation remains MSE because the final anomaly
        # score is also MSE.
        # ----------------------------------------------------

        model.eval()

        validation_errors = []

        for path in val_files:

            error = recording_error(
                model,
                path
            )

            validation_errors.append(
                error
            )

        val_loss = float(
            np.mean(
                validation_errors
            )
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs} | "
            f"Train Loss: "
            f"{train_loss:.8f} | "
            f"Val MSE: "
            f"{val_loss:.8f} | "
            f"LR: "
            f"{current_lr:.2e}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key:
                value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict()
                .items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./mimii_data"
    )

    parser.add_argument(
        "--machine",
        type=str,
        default="fan"
    )

    parser.add_argument(
        "--machine_id",
        type=str,
        default="id_00"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 70)
    print("2D CONVOLUTIONAL AUTOENCODER")
    print("MIMII ANOMALY DETECTION")
    print("COMPOSITE MSE + TIME/FREQUENCY GRADIENT LOSS")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Machine: {args.machine}"
    )

    print(
        f"Machine ID: {args.machine_id}"
    )

    # ========================================================
    # FIND FILES
    # ========================================================

    normal_files, abnormal_files = find_files(
        args.data_root,
        args.machine,
        args.machine_id
    )

    print()

    print(
        f"Normal recordings: "
        f"{len(normal_files)}"
    )

    print(
        f"Abnormal recordings: "
        f"{len(abnormal_files)}"
    )

    if len(normal_files) == 0:

        raise RuntimeError(
            "No normal WAV files found."
        )

    if len(abnormal_files) == 0:

        raise RuntimeError(
            "No abnormal WAV files found."
        )

    # ========================================================
    # RECORDING-LEVEL SPLIT
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
        f"Training normal recordings: "
        f"{len(train_files)}"
    )

    print(
        f"Validation normal recordings: "
        f"{len(validation_files)}"
    )

    print(
        f"Normal test recordings: "
        f"{len(normal_test_files)}"
    )

    print(
        f"Abnormal test recordings: "
        f"{len(abnormal_files)}"
    )

    # ========================================================
    # TRAINING DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING TRAINING DATASET")
    print("=" * 70)

    train_dataset = MelWindowDataset(
        train_files
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING 2D-CAE")
    print("=" * 70)

    model = ConvAutoencoder().to(
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
        model=model,
        train_loader=train_loader,
        val_files=validation_files,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    # ========================================================
    # VALIDATION ERRORS
    # ========================================================

    print()
    print("=" * 70)
    print("CALCULATING VALIDATION ERRORS")
    print("=" * 70)

    validation_errors = []

    for index, path in enumerate(
        validation_files
    ):

        error = recording_error(
            model,
            path
        )

        validation_errors.append(
            error
        )

        if (
            index + 1
        ) % 50 == 0:

            print(
                f"Validation: "
                f"{index + 1}/"
                f"{len(validation_files)}"
            )

    validation_errors = np.array(
        validation_errors
    )

    mean_error = float(
        np.mean(
            validation_errors
        )
    )

    std_error = float(
        np.std(
            validation_errors
        )
    )

    threshold = (
        mean_error
        + 3.0 * std_error
    )

    print()

    print(
        f"Validation mean MSE: "
        f"{mean_error:.8f}"
    )

    print(
        f"Validation std MSE: "
        f"{std_error:.8f}"
    )

    print(
        f"Anomaly threshold "
        f"(μ + 3σ): "
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

    true_labels = (
        [0] * len(normal_test_files)
        + [1] * len(abnormal_files)
    )

    test_errors = []

    for index, path in enumerate(
        test_files
    ):

        error = recording_error(
            model,
            path
        )

        test_errors.append(
            error
        )

        if (
            index + 1
        ) % 50 == 0:

            print(
                f"Testing: "
                f"{index + 1}/"
                f"{len(test_files)}"
            )

    test_errors = np.array(
        test_errors
    )

    predictions = (
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
            true_labels,
            predictions,
            target_names=[
                "Normal",
                "Anomalous"
            ],
            digits=4
        )
    )

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    try:

        auc = roc_auc_score(
            true_labels,
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
    # SAVE MODEL
    # ========================================================

    model_name = (
        f"paper_2d_cae_"
        f"{args.machine}_"
        f"{args.machine_id}_"
        f"GRADIENT.pt"
    )

    torch.save(
        model.state_dict(),
        model_name
    )

    print()

    print(
        f"Model saved to: "
        f"{model_name}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    report = classification_report(
        true_labels,
        predictions,
        target_names=[
            "Normal",
            "Anomalous"
        ],
        output_dict=True
    )

    results = {

        "machine": args.machine,

        "machine_id": args.machine_id,

        "sample_rate": SAMPLE_RATE,

        "n_fft": N_FFT,

        "hop_length": HOP_LENGTH,

        "n_mels": N_MELS,

        "window_seconds": WINDOW_SECONDS,

        "window_overlap": 0.50,

        "validation_fraction": VAL_FRACTION,

        "training_loss":
            "MSE + 0.20 * time_frequency_gradient_L1",

        "anomaly_score":
            "recording_mean_window_MSE",

        "threshold_method":
            "mean + 3*std",

        "validation_mean_error":
            mean_error,

        "validation_std_error":
            std_error,

        "threshold":
            threshold,

        "roc_auc":
            auc,

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,

        "train_recordings":
            len(train_files),

        "validation_recordings":
            len(validation_files),

        "normal_test_recordings":
            len(normal_test_files),

        "abnormal_test_recordings":
            len(abnormal_files),

        "epochs_requested":
            args.epochs,

        "batch_size":
            args.batch,

        "learning_rate":
            args.lr
    }

    result_name = (
        f"paper_2d_cae_"
        f"{args.machine}_"
        f"{args.machine_id}_"
        f"GRADIENT_results.json"
    )

    with open(
        result_name,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"Results saved to: "
        f"{result_name}"
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