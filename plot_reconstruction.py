import numpy as np
import matplotlib.pyplot as plt
import torch

from mimii_pipeline import (
    ConvAutoencoder2D,
    wav_to_fixed_spec,
    find_mimii_files
)

DATA_ROOT = "./synthetic_data"
MACHINE = "fan"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normal_files, abnormal_files = find_mimii_files(
    DATA_ROOT,
    MACHINE
)

# Select one normal audio file
audio_file = normal_files[0]

# Convert audio to Mel-spectrogram
original_spec = wav_to_fixed_spec(audio_file)

# Load trained model
model = ConvAutoencoder2D().to(device)

model.load_state_dict(
    torch.load(
        f"autoencoder_{MACHINE}.pt",
        map_location=device
    )
)

model.eval()

# Convert spectrogram to PyTorch tensor
input_tensor = torch.tensor(
    original_spec,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

# Reconstruct spectrogram
with torch.no_grad():
    reconstructed = model(input_tensor)

reconstructed_spec = reconstructed.squeeze().cpu().numpy()

# Calculate reconstruction error
mse = np.mean(
    (original_spec - reconstructed_spec) ** 2
)

print("Audio file:", audio_file)
print("Reconstruction MSE:", mse)

# Plot original spectrogram
plt.figure(figsize=(10, 5))

plt.imshow(
    original_spec,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Normalized Power")
plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency")
plt.title("Original Mel-Spectrogram")

plt.tight_layout()
plt.savefig(
    "original_mel_spectrogram.png",
    dpi=300
)

plt.show()

# Plot reconstructed spectrogram
plt.figure(figsize=(10, 5))

plt.imshow(
    reconstructed_spec,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Normalized Power")
plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency")
plt.title("Reconstructed Mel-Spectrogram")

plt.tight_layout()
plt.savefig(
    "reconstructed_mel_spectrogram.png",
    dpi=300
)

plt.show()

print("Original spectrogram saved as original_mel_spectrogram.png")
print("Reconstructed spectrogram saved as reconstructed_mel_spectrogram.png")