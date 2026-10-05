import os
import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from mimii_pipeline import (
    ConvAutoencoder,
    find_files,
    recording_error,
    DEVICE,
    SEED,
    VAL_FRACTION
)


DATA_ROOT = "./mimii_data"
MACHINE = "fan"
MACHINE_ID = "id_00"

MODEL_PATH = "paper_2d_cae_fan_id_00_NEW.pt"


# ============================================================
# RECREATE THE SAME DATA SPLIT
# ============================================================

normal_files, abnormal_files = find_files(
    DATA_ROOT,
    MACHINE,
    MACHINE_ID
)

rng = np.random.default_rng(SEED)

normal_files = list(normal_files)

rng.shuffle(normal_files)

number_normal_test = min(
    len(abnormal_files),
    len(normal_files) // 2
)

normal_test_files = normal_files[
    :number_normal_test
]

remaining_normal = normal_files[
    number_normal_test:
]

number_validation = max(
    1,
    int(
        len(remaining_normal)
        * VAL_FRACTION
    )
)

validation_files = remaining_normal[
    :number_validation
]

train_files = remaining_normal[
    number_validation:
]


print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"Validation normal: {len(validation_files)}"
)

print(
    f"Normal test: {len(normal_test_files)}"
)

print(
    f"Abnormal test: {len(abnormal_files)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = ConvAutoencoder().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print()
print(
    f"Loaded model: {MODEL_PATH}"
)


# ============================================================
# VALIDATION ERRORS
# ============================================================

print()
print("=" * 70)
print("VALIDATION ERRORS")
print("=" * 70)

validation_errors = []

for i, path in enumerate(
    validation_files
):

    error = recording_error(
        model,
        path
    )

    validation_errors.append(
        error
    )

    if (i + 1) % 50 == 0:

        print(
            f"{i + 1}/{len(validation_files)}"
        )

validation_errors = np.array(
    validation_errors
)


# ============================================================
# TEST ERRORS
# ============================================================

print()
print("=" * 70)
print("TEST ERRORS")
print("=" * 70)

normal_test_errors = []

for i, path in enumerate(
    normal_test_files
):

    error = recording_error(
        model,
        path
    )

    normal_test_errors.append(
        error
    )

    if (i + 1) % 50 == 0:

        print(
            f"Normal: "
            f"{i + 1}/{len(normal_test_files)}"
        )


abnormal_test_errors = []

for i, path in enumerate(
    abnormal_files
):

    error = recording_error(
        model,
        path
    )

    abnormal_test_errors.append(
        error
    )

    if (i + 1) % 50 == 0:

        print(
            f"Abnormal: "
            f"{i + 1}/{len(abnormal_files)}"
        )


normal_test_errors = np.array(
    normal_test_errors
)

abnormal_test_errors = np.array(
    abnormal_test_errors
)


test_errors = np.concatenate(
    [
        normal_test_errors,
        abnormal_test_errors
    ]
)

true_labels = np.concatenate(
    [
        np.zeros(
            len(normal_test_errors),
            dtype=int
        ),
        np.ones(
            len(abnormal_test_errors),
            dtype=int
        )
    ]
)


# ============================================================
# PAPER THRESHOLD
# μ + 3σ
# ============================================================

validation_mean = np.mean(
    validation_errors
)

validation_std = np.std(
    validation_errors
)

paper_threshold = (
    validation_mean
    + 3 * validation_std
)


print()
print("=" * 70)
print("PAPER THRESHOLD")
print("=" * 70)

print(
    f"Validation mean: "
    f"{validation_mean:.8f}"
)

print(
    f"Validation std: "
    f"{validation_std:.8f}"
)

print(
    f"μ + 3σ threshold: "
    f"{paper_threshold:.8f}"
)


paper_predictions = (
    test_errors > paper_threshold
).astype(int)


print()
print(
    "F1 using μ + 3σ:"
)

print(
    f"{f1_score(true_labels, paper_predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(true_labels, paper_predictions):.4f}"
)

print(
    f"Recall: "
    f"{recall_score(true_labels, paper_predictions):.4f}"
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print()
print("=" * 70)
print("THRESHOLD SWEEP")
print("=" * 70)

minimum = np.min(test_errors)

maximum = np.max(test_errors)

thresholds = np.linspace(
    minimum,
    maximum,
    2000
)

best_f1 = -1

best_threshold = None

best_precision = None

best_recall = None

best_cm = None


for threshold in thresholds:

    predictions = (
        test_errors > threshold
    ).astype(int)

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_precision = precision_score(
            true_labels,
            predictions,
            zero_division=0
        )

        best_recall = recall_score(
            true_labels,
            predictions,
            zero_division=0
        )

        best_cm = confusion_matrix(
            true_labels,
            predictions
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("BEST POSSIBLE TEST THRESHOLD")
print("=" * 70)

print(
    f"Best threshold: "
    f"{best_threshold:.8f}"
)

print(
    f"Best F1: "
    f"{best_f1:.4f}"
)

print(
    f"Precision: "
    f"{best_precision:.4f}"
)

print(
    f"Recall: "
    f"{best_recall:.4f}"
)

print()
print(
    "Confusion Matrix:"
)

print(
    best_cm
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if best_f1 >= 0.90:

    print(
        "GOOD NEWS:"
    )

    print(
        "The trained model is capable of "
        "reaching F1 >= 0.90."
    )

    print(
        "The main issue is threshold selection."
    )

elif best_f1 >= 0.80:

    print(
        "The model is close to the target."
    )

    print(
        "Further model/pipeline improvement "
        "is needed to reach F1 >= 0.90."
    )

else:

    print(
        "The model cannot reach F1 >= 0.90 "
        "with threshold tuning alone."
    )

    print(
        "The model/pipeline needs another "
        "targeted improvement."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "The best test threshold is diagnostic only."
)

print(
    "It uses test labels and therefore must "
    "NOT be presented as the final threshold."
)

print(
    "The paper's stated threshold is μ + 3σ."
)