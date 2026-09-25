import os
import numpy as np
import soundfile as sf

SR = 16000
DURATION = 1
SAMPLES = SR * DURATION

BASE_DIR = "synthetic_data"
NORMAL_DIR = os.path.join(BASE_DIR, "fan", "id_00", "normal")
ABNORMAL_DIR = os.path.join(BASE_DIR, "fan", "id_00", "abnormal")

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ABNORMAL_DIR, exist_ok=True)

np.random.seed(42)


def generate_normal():
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)

    signal = (
        0.55 * np.sin(2 * np.pi * 300 * t) +
        0.25 * np.sin(2 * np.pi * 600 * t) +
        0.12 * np.sin(2 * np.pi * 900 * t)
    )

    noise = 0.015 * np.random.randn(SAMPLES)

    return signal + noise


def generate_abnormal():
    t = np.linspace(0, DURATION, SAMPLES, endpoint=False)

    signal = (
        0.55 * np.sin(2 * np.pi * 300 * t) +
        0.25 * np.sin(2 * np.pi * 600 * t) +
        0.12 * np.sin(2 * np.pi * 900 * t)
    )

    # Fault-like frequency component
    fault_frequency = 2200
    fault_signal = 0.45 * np.sin(2 * np.pi * fault_frequency * t)

    # Irregular amplitude modulation
    modulation = 1 + 0.7 * np.sin(2 * np.pi * 7 * t)

    # Strong irregular noise
    noise = 0.12 * np.random.randn(SAMPLES)

    # Short impact/transient
    impact = np.zeros(SAMPLES)
    impact_start = int(0.35 * SR)
    impact_length = int(0.02 * SR)

    impact[impact_start:impact_start + impact_length] = (
        0.8 * np.sin(
            2 * np.pi * 3000 *
            np.arange(impact_length) / SR
        )
    )

    return signal + (fault_signal * modulation) + noise + impact


for i in range(100):
    audio = generate_normal()

    sf.write(
        os.path.join(NORMAL_DIR, f"normal_{i:03d}.wav"),
        audio,
        SR
    )


for i in range(50):
    audio = generate_abnormal()

    sf.write(
        os.path.join(ABNORMAL_DIR, f"abnormal_{i:03d}.wav"),
        audio,
        SR
    )


print("Synthetic dataset regenerated successfully!")
print("Normal samples   :", 100)
print("Abnormal samples :", 50)
print("Location         :", BASE_DIR)