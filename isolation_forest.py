import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from mimii_pipeline import find_mimii_files

DATA_ROOT = "./synthetic_data"
MACHINE = "fan"
SR = 16000


def extract_features(path):
    audio, _ = librosa.load(path, sr=SR)

    rms = np.mean(librosa.feature.rms(y=audio))

    mean = np.mean(audio)
    std = np.std(audio)

    if std == 0:
        skewness = 0
        kurtosis = 0
    else:
        normalized = (audio - mean) / std
        skewness = np.mean(normalized ** 3)
        kurtosis = np.mean(normalized ** 4)

    peak = np.max(np.abs(audio))

    if rms == 0:
        crest_factor = 0
    else:
        crest_factor = peak / rms

    return [
        rms,
        skewness,
        kurtosis,
        crest_factor
    ]


normal_files, abnormal_files = find_mimii_files(
    DATA_ROOT,
    MACHINE
)

rng = np.random.RandomState(42)
rng.shuffle(normal_files)

n_val = max(1, int(len(normal_files) * 0.15))

val_files = normal_files[:n_val]
train_files = normal_files[n_val:]


print("Extracting features from normal training data...")

train_features = np.array([
    extract_features(path)
    for path in train_files
])

print("Extracting validation features...")

val_features = np.array([
    extract_features(path)
    for path in val_files
])

print("Extracting abnormal features...")

abnormal_features = np.array([
    extract_features(path)
    for path in abnormal_files
])


model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(train_features)


test_features = np.concatenate(
    [val_features, abnormal_features],
    axis=0
)

true_labels = np.array(
    [0] * len(val_features) +
    [1] * len(abnormal_features)
)


predictions = model.predict(test_features)

predictions = np.where(
    predictions == -1,
    1,
    0
)


print("\n================ Isolation Forest ================")

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
print(confusion_matrix(true_labels, predictions))


scores = -model.decision_function(test_features)

try:
    auc = roc_auc_score(true_labels, scores)
    print(f"ROC-AUC: {auc:.4f}")
except ValueError:
    pass