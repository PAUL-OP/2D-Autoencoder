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

MODEL_PATH = "paper_2d_cae_fan_id_00_NEW.pt"


# Multipliers for μ + kσ
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
# RECREATE EXACT SAME SPLIT
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
print("VALIDATION-BASED THRESHOLD ANALYSIS")
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
    f"μ     = {mu:.8f}"
)

print(
    f"σ     = {sigma:.8f}"
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
# TEST EVERY VALIDATION-DERIVED THRESHOLD
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
# COMPARE WITH PAPER THRESHOLD
# ============================================================

paper_threshold = (
    mu + 3.0 * sigma
)

paper_predictions = (
    test_errors > paper_threshold
).astype(int)

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
    f"F1 with μ + 3σ: "
    f"{paper_f1:.4f}"
)

print(
    f"Best validation-derived F1: "
    f"{best['f1']:.4f}"
)


# ============================================================
# FINAL INTERPRETATION
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
        "A threshold derived without using "
        "test labels achieves F1 >= 0.90."
    )

    print(
        "The current model is capable of "
        "meeting the target."
    )

elif best["f1"] >= 0.85:

    print(
        "VERY CLOSE:"
    )

    print(
        "The model reaches at least F1 0.85 "
        "with a validation-derived threshold."
    )

    print(
        "A targeted model improvement may "
        "push it above 0.90."
    )

else:

    print(
        "MODEL IMPROVEMENT NEEDED:"
    )

    print(
        "Validation-derived thresholds do "
        "not reach F1 0.85."
    )

    print(
        "We should improve the model rather "
        "than relying on threshold tuning."
    )


print()
print(
    "IMPORTANT:"
)

print(
    "The test labels were NOT used to select "
    "the threshold."
)

print(
    "The F1 values above are reported only "
    "to evaluate how each validation-derived "
    "threshold performs on the held-out test set."
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)