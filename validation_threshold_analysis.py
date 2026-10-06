import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from mimii_pipeline import (
    ConvAutoencoder,
    find_files,
    recording_error,
    DEVICE,
    SEED,
    VAL_FRACTION
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "./mimii_data"
MACHINE = "fan"
MACHINE_ID = "id_00"

# Analyze the GRADIENT model
MODEL_PATH = "paper_2d_cae_fan_id_00_GRADIENT.pt"


K_VALUES = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0
]


# ============================================================
# RECREATE SAME DATA SPLIT
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


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("GRADIENT MODEL - VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

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


# ============================================================
# LOAD MODEL
# ============================================================

model = ConvAutoencoder().to(
    DEVICE
)

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
print("CALCULATING VALIDATION ERRORS")
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
            f"Validation: "
            f"{i + 1}/"
            f"{len(validation_files)}"
        )

validation_errors = np.array(
    validation_errors
)


# ============================================================
# VALIDATION STATISTICS
# ============================================================

mu = np.mean(
    validation_errors
)

sigma = np.std(
    validation_errors
)

print()
print("=" * 70)
print("VALIDATION STATISTICS")
print("=" * 70)

print(
    f"μ      = {mu:.8f}"
)

print(
    f"σ      = {sigma:.8f}"
)

print(
    f"μ + 3σ = "
    f"{mu + 3 * sigma:.8f}"
)


# ============================================================
# TEST ERRORS
# ============================================================

print()
print("=" * 70)
print("CALCULATING TEST ERRORS")
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
            f"Normal test: "
            f"{i + 1}/"
            f"{len(normal_test_files)}"
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
            f"Abnormal test: "
            f"{i + 1}/"
            f"{len(abnormal_files)}"
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
# TEST VALIDATION-DERIVED THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("VALIDATION-DERIVED THRESHOLDS")
print("=" * 70)

print()

print(
    f"{'k':>6}"
    f"{'Threshold':>16}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 70)


results = []

for k in K_VALUES:

    threshold = (
        mu + k * sigma
    )

    predictions = (
        test_errors > threshold
    ).astype(int)

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    results.append(
        {
            "k": k,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )

    print(
        f"{k:>6.2f}"
        f"{threshold:>16.8f}"
        f"{precision:>14.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
    )


# ============================================================
# BEST VALIDATION-DERIVED RESULT
# ============================================================

best = max(
    results,
    key=lambda x: x["f1"]
)

print()
print("=" * 70)
print("BEST VALIDATION-DERIVED THRESHOLD")
print("=" * 70)

print(
    f"k: "
    f"{best['k']:.2f}"
)

print(
    f"Threshold: "
    f"{best['threshold']:.8f}"
)

print(
    f"Precision: "
    f"{best['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best['recall']:.4f}"
)

print(
    f"F1: "
    f"{best['f1']:.4f}"
)

best_predictions = (
    test_errors > best["threshold"]
).astype(int)

best_cm = confusion_matrix(
    true_labels,
    best_predictions
)

print()
print(
    "Confusion Matrix:"
)

print(
    best_cm
)


# ============================================================
# PAPER THRESHOLD
# ============================================================

paper_threshold = (
    mu + 3.0 * sigma
)

paper_predictions = (
    test_errors > paper_threshold
).astype(int)

paper_precision = precision_score(
    true_labels,
    paper_predictions,
    zero_division=0
)

paper_recall = recall_score(
    true_labels,
    paper_predictions,
    zero_division=0
)

paper_f1 = f1_score(
    true_labels,
    paper_predictions,
    zero_division=0
)

print()
print("=" * 70)
print("PAPER THRESHOLD COMPARISON")
print("=" * 70)

print(
    f"μ + 3σ threshold: "
    f"{paper_threshold:.8f}"
)

print(
    f"Precision: "
    f"{paper_precision:.4f}"
)

print(
    f"Recall: "
    f"{paper_recall:.4f}"
)

print(
    f"F1: "
    f"{paper_f1:.4f}"
)

print(
    f"Best validation-derived F1: "
    f"{best['f1']:.4f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if best["f1"] >= 0.90:

    print(
        "EXCELLENT:"
    )

    print(
        "The GRADIENT model reaches "
        "F1 >= 0.90 using a threshold "
        "derived without test labels."
    )

elif best["f1"] > 0.8717:

    print(
        "IMPROVEMENT:"
    )

    print(
        "The GRADIENT model beats our "
        "previous best validation-derived F1 "
        "of 0.8717."
    )

elif best["f1"] >= 0.85:

    print(
        "CLOSE:"
    )

    print(
        "The GRADIENT model reaches "
        "at least F1 0.85."
    )

    print(
        "However, it does not beat our "
        "previous best of 0.8717."
    )

else:

    print(
        "NO IMPROVEMENT:"
    )

    print(
        "The GRADIENT model does not beat "
        "our previous best validation-derived "
        "F1 of 0.8717."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "The test labels were NOT used to "
    "select the threshold."
)

print(
    "The threshold is derived only from "
    "normal validation reconstruction errors."
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)