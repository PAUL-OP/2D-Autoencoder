"""
Anomalous Sound Detection Pipeline — MIMII Dataset Edition
============================================================
Trains a 2D convolutional autoencoder on NORMAL machine-operating sounds
from the MIMII dataset (Purohit et al., 2019, Hitachi) and flags anomalies
via reconstruction error. Same core architecture as the original synthetic
pipeline, but now reads real 10-second .wav recordings from disk.

--------------------------------------------------------------------------
DATASET SETUP (one-time, do this on your own machine before running this
script — Zenodo isn't reachable from every sandboxed environment):

1. Download the MIMII zip(s) you want from Zenodo:
       https://zenodo.org/record/3384388
   Files are named like "6_dB_fan.zip", "0_dB_pump.zip", "-6_dB_valve.zip".
   Pick one SNR level and one machine type to start (fan is the easiest).
   (`download_mimii.py`, included alongside this file, automates this step.)

2. Unzip it. You should end up with a layout like:

       <data_root>/
           fan/
               id_00/
                   normal/*.wav
                   abnormal/*.wav
               id_02/
                   normal/*.wav
                   abnormal/*.wav
               id_04/...
               id_06/...
           pump/... (if you downloaded pump too)

   This is the standard MIMII layout: <data_root>/<machine>/id_XX/{normal,abnormal}/*.wav

3. Run:
       python mimii_pipeline.py --data_root /path/to/data_root --machine fan

--------------------------------------------------------------------------
"""
import os
import glob
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Feature Extraction: Audio -> Mel-Spectrogram
# ==========================================
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FIXED_FRAMES = 128  # time axis is padded/cropped to this so the model's shape stays fixed
                     # regardless of each clip's exact length


def audio_to_mel_spectrogram(audio, sr=16000, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Converts a 1D raw audio array into a normalized 2D Mel-Spectrogram matrix."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    norm_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    return norm_spec


def pad_or_crop(spec, frames=FIXED_FRAMES):
    """Force the time axis to a fixed number of frames: center-crop if longer,
    zero-pad on the right if shorter. Keeps every spectrogram the same shape
    so it can be batched and fed through a fixed-size conv net."""
    t = spec.shape[1]
    if t == frames:
        return spec
    if t > frames:
        start = (t - frames) // 2
        return spec[:, start:start + frames]
    pad_width = frames - t
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")


def wav_to_fixed_spec(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    spec = audio_to_mel_spectrogram(audio, sr=sr)
    return pad_or_crop(spec)


class AudioDataset(Dataset):
    def __init__(self, specs):
        # (N, H, W) -> (N, 1, H, W)
        self.specs = torch.tensor(np.array(specs), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]


# ==========================================
# 2. Model Architecture: 2D Convolutional Autoencoder
#    (1, 128, 128) -> (64, 16, 16) -> (1, 128, 128)
# ==========================================
class ConvAutoencoder2D(nn.Module):
    def __init__(self):
        super(ConvAutoencoder2D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (1, 128, 128)
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# ==========================================
# 3. Real Dataset Loader (MIMII directory layout)
# ==========================================
def find_mimii_files(data_root, machine, machine_ids=None):
    """
    Expects: <data_root>/<machine>/id_XX/{normal,abnormal}/*.wav
    Returns two lists of file paths (not loaded into memory yet).
    """
    machine_dir = os.path.join(data_root, machine)
    if not os.path.isdir(machine_dir):
        raise FileNotFoundError(
            f"Could not find '{machine_dir}'.\n"
            f"Expected the MIMII layout: <data_root>/<machine>/id_XX/{{normal,abnormal}}/*.wav\n"
            f"Check --data_root and --machine, and that you unzipped the download."
        )

    id_dirs = sorted(glob.glob(os.path.join(machine_dir, "id_*")))
    if machine_ids:
        id_dirs = [d for d in id_dirs if os.path.basename(d) in machine_ids]

    normal_files, abnormal_files = [], []
    for id_dir in id_dirs:
        normal_files += sorted(glob.glob(os.path.join(id_dir, "normal", "*.wav")))
        abnormal_files += sorted(glob.glob(os.path.join(id_dir, "abnormal", "*.wav")))

    if not normal_files:
        raise FileNotFoundError(f"No normal/*.wav files found under {machine_dir}")

    return normal_files, abnormal_files


def specs_from_files(file_list, sr=16000, desc=""):
    specs = []
    for i, path in enumerate(file_list):
        specs.append(wav_to_fixed_spec(path, sr=sr))
        if (i + 1) % 200 == 0:
            print(f"  [{desc}] processed {i + 1}/{len(file_list)}")
    return np.array(specs)


# ==========================================
# 4. Main Training and Evaluation Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train an anomalous-sound-detection autoencoder on MIMII data")
    parser.add_argument("--data_root", required=True,
                         help="Path to the unzipped MIMII folder that CONTAINS the <machine> subfolder")
    parser.add_argument("--machine", default="fan", choices=["fan", "pump", "slider", "valve"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15,
                         help="Fraction of normal files held out to calibrate the anomaly threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Discover real audio files
    print(f"\nScanning MIMII files for machine='{args.machine}' under {args.data_root} ...")
    normal_files, abnormal_files = find_mimii_files(args.data_root, args.machine)
    print(f"Found {len(normal_files)} normal / {len(abnormal_files)} abnormal recordings.")

    rng = np.random.RandomState(42)
    rng.shuffle(normal_files)
    n_val = max(1, int(len(normal_files) * args.val_frac))
    val_files = normal_files[:n_val]
    train_files = normal_files[n_val:]

    # 2. Extract Mel-Spectrogram features from real audio
    print("\nExtracting Mel-Spectrograms from audio...")
    train_specs = specs_from_files(train_files, desc="train")
    val_specs = specs_from_files(val_files, desc="val")
    abn_specs = specs_from_files(abnormal_files, desc="abnormal")

    test_specs = np.concatenate([val_specs, abn_specs], axis=0)
    test_labels = np.array([0] * len(val_specs) + [1] * len(abn_specs))  # 0: Normal, 1: Anomalous

    train_loader = DataLoader(AudioDataset(train_specs), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_specs), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(AudioDataset(test_specs), batch_size=1, shuffle=False)

    # 3. Instantiate Model, Loss, Optimizer
    model = ConvAutoencoder2D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Train Model (strictly on normal data, unsupervised)
    print("\nStarting Training (Unsupervised, normal sounds only)...")
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        epoch_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {epoch_loss:.6f}")

    # 5. Calculate Decision Threshold (mu + 3 * sigma rule) from held-out NORMAL data
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss_per_sample = torch.mean((recon - batch) ** 2, dim=[1, 2, 3])
            val_losses.extend(loss_per_sample.cpu().numpy())

    mu_loss = np.mean(val_losses)
    sigma_loss = np.std(val_losses)
    threshold = mu_loss + (3 * sigma_loss)
    print(f"\nCalculated Anomaly Threshold (mu + 3*sigma): {threshold:.6f}")

    # 6. Inference & Anomaly Flagging
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = torch.mean((recon - batch) ** 2).item()
            test_losses.append(loss)

    test_losses = np.array(test_losses)
    predictions = (test_losses > threshold).astype(int)

    # 7. Print Evaluation Metrics
    print("\n================ Classification Report ================")
    print(classification_report(test_labels, predictions, target_names=["Normal (0)", "Anomalous (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    try:
        auc = roc_auc_score(test_labels, test_losses)
        print(f"ROC-AUC (raw reconstruction error, threshold-independent): {auc:.4f}")
    except ValueError:
        pass  # only one class present in test set

    # 8. Save the trained model
    save_path = f"autoencoder_{args.machine}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
