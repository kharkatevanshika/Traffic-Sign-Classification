import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = "../model/traffic_sign_cnn.keras"

X_VAL_FILE = "../model/X_val.npy"
Y_VAL_FILE = "../model/y_val.npy"
CLASS_NAMES_FILE = "../model/class_names.pkl"

RESULTS_DIR = "../model/evaluation_results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PHASE 4 - MODEL EVALUATION")
print("=" * 60)

print("\nLoading validation data...")

X_val = np.load(X_VAL_FILE)
y_val = np.load(Y_VAL_FILE)

print("Validation images:", X_val.shape)
print("Validation labels:", y_val.shape)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_FILE, "rb") as f:
    class_names = pickle.load(f)

print("Number of classes:", len(class_names))


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained CNN model...")

model = load_model(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_val, y_val, verbose=1)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Validation Loss     : {loss:.6f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_val, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ============================================================
# ACCURACY
# ============================================================

final_accuracy = accuracy_score(y_val, y_pred)

print("\nPrediction Accuracy:")
print(f"{final_accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_val,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=[str(name) for name in class_names],
    zero_division=0
)

print(report)

# Save classification report
with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("TRAFFIC SIGN CLASSIFICATION - CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=np.arange(len(class_names))
)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Traffic Sign Classification - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

confusion_matrix_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_matrix_file, dpi=300)

plt.show()

print("Confusion matrix saved at:")
print(confusion_matrix_file)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(class_names):

    total = np.sum(y_val == i)

    correct = np.sum(
        (y_val == i) & (y_pred == i)
    )

    if total > 0:
        class_accuracy = (correct / total) * 100
    else:
        class_accuracy = 0

    print(
        f"Class {i:2d} ({class_name}) : "
        f"{class_accuracy:.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    RESULTS_DIR,
    "evaluation_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("TRAFFIC SIGN CLASSIFICATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Number of Classes: {len(class_names)}\n")
    f.write(f"Validation Images: {len(X_val)}\n")
    f.write(f"Validation Loss: {loss:.6f}\n")
    f.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Prediction Accuracy: {final_accuracy * 100:.2f}%\n")


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PHASE 4 EVALUATION COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print("1. classification_report.txt")
print("2. confusion_matrix.png")
print("3. evaluation_summary.txt")

print("\nAll files saved inside:")
print(RESULTS_DIR)