import os
import glob
import random

import numpy as np
import librosa

from sklearn.ensemble import IsolationForest
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

SEED = 42

# Same split used by our 2D-CAE experiments
VAL_FRACTION = 0.15


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# FIND FILES
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
# STATISTICAL FEATURES
#
# Paper:
# RMS
# Kurtosis
# Skewness
# Crest Factor
# ============================================================

def extract_features(path):

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(
        np.float64
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            audio ** 2
        )
    )

    # --------------------------------------------------------
    # Mean and standard deviation
    # --------------------------------------------------------

    mean = np.mean(
        audio
    )

    centered = (
        audio - mean
    )

    std = np.std(
        audio
    )

    # Avoid division by zero
    if std < 1e-12:

        skewness = 0.0
        kurtosis = 0.0

    else:

        # ----------------------------------------------------
        # Skewness
        # ----------------------------------------------------

        skewness = (
            np.mean(
                centered ** 3
            )
            / (std ** 3)
        )

        # ----------------------------------------------------
        # Kurtosis
        #
        # Pearson kurtosis:
        # E[(x-mu)^4] / sigma^4
        # ----------------------------------------------------

        kurtosis = (
            np.mean(
                centered ** 4
            )
            / (std ** 4)
        )

    # --------------------------------------------------------
    # Crest factor
    #
    # peak amplitude / RMS
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if rms < 1e-12:

        crest_factor = 0.0

    else:

        crest_factor = (
            peak / rms
        )

    return np.array(
        [
            rms,
            kurtosis,
            skewness,
            crest_factor
        ],
        dtype=np.float64
    )


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_features(
    files,
    name
):

    features = []

    total = len(files)

    print()

    print(
        f"Extracting features from "
        f"{name}: {total} recordings"
    )

    for i, path in enumerate(
        files
    ):

        feature_vector = extract_features(
            path
        )

        features.append(
            feature_vector
        )

        if (
            (i + 1) % 50 == 0
            or i + 1 == total
        ):

            print(
                f"{i + 1}/{total}"
            )

    return np.vstack(
        features
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISOLATION FOREST")
    print("MIMII FAN ID_00")
    print("=" * 70)

    # ========================================================
    # FIND DATA
    # ========================================================

    normal_files, abnormal_files = find_files(
        DATA_ROOT,
        MACHINE,
        MACHINE_ID
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

    # Equal normal and abnormal test set
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

    # Validation
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

    # Training
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
        f"Normal training: "
        f"{len(train_files)}"
    )

    print(
        f"Normal validation: "
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
    # FEATURE EXTRACTION
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print(
        "Features:"
    )

    print(
        "1. RMS"
    )

    print(
        "2. Kurtosis"
    )

    print(
        "3. Skewness"
    )

    print(
        "4. Crest Factor"
    )

    X_train = build_features(
        train_files,
        "NORMAL TRAINING"
    )

    X_validation = build_features(
        validation_files,
        "NORMAL VALIDATION"
    )

    X_normal_test = build_features(
        normal_test_files,
        "NORMAL TEST"
    )

    X_abnormal_test = build_features(
        abnormal_files,
        "ABNORMAL TEST"
    )

    # ========================================================
    # SHOW FEATURE SHAPES
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE MATRICES")
    print("=" * 70)

    print(
        f"Training: "
        f"{X_train.shape}"
    )

    print(
        f"Validation: "
        f"{X_validation.shape}"
    )

    print(
        f"Normal test: "
        f"{X_normal_test.shape}"
    )

    print(
        f"Abnormal test: "
        f"{X_abnormal_test.shape}"
    )

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )

    # Train ONLY on normal data
    model.fit(
        X_train
    )

    print(
        "Isolation Forest trained "
        "using normal recordings only."
    )

    # ========================================================
    # TEST
    # ========================================================

    X_test = np.vstack(
        [
            X_normal_test,
            X_abnormal_test
        ]
    )

    y_test = np.concatenate(
        [
            np.zeros(
                len(X_normal_test),
                dtype=int
            ),
            np.ones(
                len(X_abnormal_test),
                dtype=int
            )
        ]
    )

    # sklearn:
    # +1 = normal
    # -1 = anomaly

    raw_predictions = model.predict(
        X_test
    )

    predictions = np.where(
        raw_predictions == -1,
        1,
        0
    )

    # ========================================================
    # ANOMALY SCORES
    #
    # Lower decision_function means
    # more anomalous.
    #
    # Negate it so that larger =
    # more anomalous.
    # ========================================================

    anomaly_scores = -model.decision_function(
        X_test
    )

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
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Anomalous"
            ],
            digits=4
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # ========================================================
    # ROC-AUC
    # ========================================================

    try:

        auc = roc_auc_score(
            y_test,
            anomaly_scores
        )

        print()

        print(
            f"ROC-AUC: "
            f"{auc:.4f}"
        )

    except Exception:

        auc = None

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURES USED")
    print("=" * 70)

    feature_names = [
        "RMS",
        "Kurtosis",
        "Skewness",
        "Crest Factor"
    ]

    for i, name in enumerate(
        feature_names
    ):

        print(
            f"{i + 1}. {name}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    import joblib

    model_path = (
        "isolation_forest_paper_id_00.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print()

    print(
        f"Model saved to: "
        f"{model_path}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print(
        "This implementation follows the paper's "
        "specified handcrafted features:"
    )

    print(
        "RMS + Kurtosis + Skewness + Crest Factor"
    )


if __name__ == "__main__":

    main()