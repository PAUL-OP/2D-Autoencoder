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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAPER / MIMII SETTINGS
# ============================================================

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = SAMPLE_RATE

# 50% overlap
OVERLAP = 0.50
STEP_SAMPLES = WINDOW_SAMPLES // 2

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# Healthy validation set for threshold
VAL_FRACTION = 0.15


# ============================================================
# MEL SPECTROGRAM
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

    mel_norm = (
        mel_db + 80.0
    ) / 80.0

    return mel_norm.astype(
        np.float32
    )


# ============================================================
# CREATE 1-SECOND OVERLAPPING WINDOWS
# ============================================================

def wav_to_windows(path):

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                WINDOW_SAMPLES - len(audio)
            )
        )

    windows = []

    for start in range(
        0,
        len(audio) - WINDOW_SAMPLES + 1,
        STEP_SAMPLES
    ):

        segment = audio[
            start:start + WINDOW_SAMPLES
        ]

        mel = audio_to_mel(
            segment
        )

        windows.append(
            mel
        )

    return windows


# ============================================================
# DATASET
# ============================================================

class MelDataset(Dataset):

    def __init__(self, files):

        self.samples = []

        print(
            "\nPreparing training spectrograms..."
        )

        for i, path in enumerate(files):

            windows = wav_to_windows(
                path
            )

            for mel in windows:

                self.samples.append(
                    mel
                )

            if (i + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{i + 1}/{len(files)} recordings"
                )

        print(
            f"Total training windows: "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        x = torch.tensor(
            self.samples[index],
            dtype=torch.float32
        )

        # [128, time]
        # ->
        # [1, 128, time]

        x = x.unsqueeze(0)

        return x


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
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

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

            nn.ReLU(),

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
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),


            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),


            nn.ConvTranspose2d(
                16,
                1,
                kernel_size=2,
                stride=2
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        encoded = self.encoder(
            x
        )

        decoded = self.decoder(
            encoded
        )

        return decoded


# ============================================================
# FIND MIMII FILES
# ============================================================

def find_files(
    data_root,
    machine,
    machine_id
):

    base = os.path.join(
        data_root,
        machine,
        machine_id
    )

    normal_dir = os.path.join(
        base,
        "normal"
    )

    abnormal_dir = os.path.join(
        base,
        "abnormal"
    )

    normal_files = sorted(
        glob.glob(
            os.path.join(
                normal_dir,
                "*.wav"
            )
        )
    )

    abnormal_files = sorted(
        glob.glob(
            os.path.join(
                abnormal_dir,
                "*.wav"
            )
        )
    )

    return (
        normal_files,
        abnormal_files
    )


# ============================================================
# RECORDING RECONSTRUCTION ERROR
# ============================================================

def recording_error(
    model,
    path
):

    windows = wav_to_windows(
        path
    )

    errors = []

    model.eval()

    with torch.no_grad():

        for mel in windows:

            x = torch.tensor(
                mel,
                dtype=torch.float32
            )

            x = x.unsqueeze(0)
            x = x.unsqueeze(0)

            x = x.to(
                DEVICE
            )

            reconstructed = model(
                x
            )

            error = torch.mean(
                (x - reconstructed) ** 2
            ).item()

            errors.append(
                error
            )

    # Average reconstruction error
    # over the entire recording.

    return float(
        np.mean(errors)
    )


# ============================================================
# CALCULATE ERRORS FOR FILES
# ============================================================

def calculate_errors(
    model,
    files,
    description
):

    errors = []

    print(
        f"\n{description}"
    )

    for i, path in enumerate(files):

        error = recording_error(
            model,
            path
        )

        errors.append(
            error
        )

        if (i + 1) % 50 == 0:

            print(
                f"{i + 1}/{len(files)}"
            )

    return np.array(
        errors,
        dtype=np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="mimii_data"
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

    args = parser.parse_args()


    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MIMII 2D-CAE PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Machine      : {args.machine}"
    )

    print(
        f"Machine ID   : {args.machine_id}"
    )

    print(
        f"Sample rate  : {SAMPLE_RATE}"
    )

    print(
        f"FFT          : {N_FFT}"
    )

    print(
        f"Hop length   : {HOP_LENGTH}"
    )

    print(
        f"Mel bands    : {N_MELS}"
    )

    print(
        f"Window       : 1 second"
    )

    print(
        f"Overlap      : 50%"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # LOAD FILES
    # ========================================================

    normal_files, abnormal_files = find_files(
        args.data_root,
        args.machine,
        args.machine_id
    )

    print(
        f"\nNormal recordings   : "
        f"{len(normal_files)}"
    )

    print(
        f"Abnormal recordings : "
        f"{len(abnormal_files)}"
    )


    if not normal_files:

        raise RuntimeError(
            "No normal WAV files found."
        )

    if not abnormal_files:

        raise RuntimeError(
            "No abnormal WAV files found."
        )


    # ========================================================
    # MIMII TEST SPLIT
    # ========================================================

    random.seed(SEED)

    shuffled_normal = normal_files.copy()

    random.shuffle(
        shuffled_normal
    )

    test_normal_count = min(
        len(abnormal_files),
        len(shuffled_normal)
    )

    test_normal_files = shuffled_normal[
        :test_normal_count
    ]

    remaining_normal = shuffled_normal[
        test_normal_count:
    ]


    # ========================================================
    # TRAIN / VALIDATION
    # ========================================================

    random.shuffle(
        remaining_normal
    )

    val_count = max(
        1,
        int(
            len(remaining_normal)
            * VAL_FRACTION
        )
    )

    val_files = remaining_normal[
        :val_count
    ]

    train_files = remaining_normal[
        val_count:
    ]


    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATA SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Training normal     : "
        f"{len(train_files)}"
    )

    print(
        f"Validation normal   : "
        f"{len(val_files)}"
    )

    print(
        f"Test normal         : "
        f"{len(test_normal_files)}"
    )

    print(
        f"Test abnormal       : "
        f"{len(abnormal_files)}"
    )


    # ========================================================
    # TRAINING DATASET
    # ========================================================

    train_dataset = MelDataset(
        train_files
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

    model = ConvAutoencoder().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()


    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )


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

            reconstructed = model(
                batch
            )

            loss = criterion(
                reconstructed,
                batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch.size(0)
            )

        epoch_loss = (
            total_loss
            / len(train_dataset)
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} "
            f"| Loss = "
            f"{epoch_loss:.8f}"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation_errors = calculate_errors(
        model,
        val_files,
        "VALIDATION"
    )


    # ========================================================
    # MU + 3 SIGMA THRESHOLD
    # ========================================================

    mean_loss = float(
        np.mean(
            validation_errors
        )
    )

    std_loss = float(
        np.std(
            validation_errors
        )
    )

    threshold = (
        mean_loss
        + 3.0 * std_loss
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "THRESHOLD"
    )

    print(
        "=" * 70
    )

    print(
        f"Mean       : "
        f"{mean_loss:.8f}"
    )

    print(
        f"Std        : "
        f"{std_loss:.8f}"
    )

    print(
        f"Threshold  : "
        f"{threshold:.8f}"
    )


    # ========================================================
    # TEST
    # ========================================================

    normal_test_errors = calculate_errors(
        model,
        test_normal_files,
        "NORMAL TEST"
    )

    abnormal_test_errors = calculate_errors(
        model,
        abnormal_files,
        "ABNORMAL TEST"
    )


    # ========================================================
    # LABELS
    # ========================================================

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_test_errors)
            ),

            np.ones(
                len(abnormal_test_errors)
            )
        ]
    )


    scores = np.concatenate(
        [
            normal_test_errors,
            abnormal_test_errors
        ]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    y_pred = (
        scores > threshold
    ).astype(int)


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Anomalous"
        ],
        digits=4
    )

    print(
        report_text
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    print(
        f"\nROC-AUC: "
        f"{roc_auc:.4f}"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        f"paper_2d_cae_"
        f"{args.machine}_"
        f"{args.machine_id}_"
        f"FINAL.pt"
    )

    torch.save(
        model.state_dict(),
        model_file
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Anomalous"
        ],
        output_dict=True
    )

    results = {

        "machine":
            args.machine,

        "machine_id":
            args.machine_id,

        "sample_rate":
            SAMPLE_RATE,

        "n_fft":
            N_FFT,

        "hop_length":
            HOP_LENGTH,

        "n_mels":
            N_MELS,

        "window_seconds":
            WINDOW_SECONDS,

        "overlap":
            OVERLAP,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "training_normal":
            len(train_files),

        "validation_normal":
            len(val_files),

        "test_normal":
            len(test_normal_files),

        "test_abnormal":
            len(abnormal_files),

        "validation_mean":
            mean_loss,

        "validation_std":
            std_loss,

        "threshold":
            threshold,

        "classification_report":
            report_dict,

        "confusion_matrix":
            cm.tolist(),

        "roc_auc":
            float(roc_auc)
    }


    results_file = (
        f"paper_2d_cae_"
        f"{args.machine}_"
        f"{args.machine_id}_"
        f"FINAL_results.json"
    )

    with open(
        results_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"Model   : {model_file}"
    )

    print(
        f"Results : {results_file}"
    )

    print(
        "\nFINAL RUN COMPLETE."
    )


if __name__ == "__main__":

    main()