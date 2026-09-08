import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# TRAFFIC SIGN CNN TRAINING
# ============================================================

print("=" * 60)
print("TRAFFIC SIGN CLASSIFICATION - CNN TRAINING")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "traffic_sign_cnn.keras"
)


# ============================================================
# CHECK PREPROCESSED DATA
# ============================================================

required_files = [
    "X_train.npy",
    "X_val.npy",
    "y_train.npy",
    "y_val.npy",
    "class_names.pkl"
]


for filename in required_files:

    file_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(file_path):

        print(
            f"\nERROR: Missing file: {filename}"
        )

        print(
            "\nRun this first:"
        )

        print(
            "python training\\preprocess.py"
        )

        raise SystemExit


# ============================================================
# DELETE BROKEN ZERO-BYTE MODEL
# ============================================================

if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    if file_size == 0:

        print(
            "\nRemoving broken 0-byte model..."
        )

        os.remove(
            MODEL_PATH
        )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed data...")


X_train = np.load(
    os.path.join(
        MODEL_DIR,
        "X_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        MODEL_DIR,
        "X_val.npy"
    )
)

y_train = np.load(
    os.path.join(
        MODEL_DIR,
        "y_train.npy"
    )

)

y_val = np.load(
    os.path.join(
        MODEL_DIR,
        "y_val.npy"
    )
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "class_names.pkl"
    ),
    "rb"
) as f:

    class_names = pickle.load(f)


NUM_CLASSES = len(
    class_names
)


# ============================================================
# DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Number of classes:",
    NUM_CLASSES
)


# ============================================================
# BUILD CNN
# ============================================================

print("\nBuilding CNN model...")


model = models.Sequential([

    layers.Input(
        shape=(32, 32, 3)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.BatchNormalization(),

    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.BatchNormalization(),

    # --------------------------------------------------------
    # CNN BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.BatchNormalization(),

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=8,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=32,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")


best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)


validation_loss, validation_accuracy = best_model.evaluate(

    X_val,

    y_val,

    verbose=0
)


print(
    f"\nValidation Loss: "
    f"{validation_loss:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.pkl"
)

with open(
    history_path,
    "wb"
) as f:

    pickle.dump(
        history.history,
        f
    )


# ============================================================
# VERIFY MODEL
# ============================================================

print("\nChecking model file...")


if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    print(
        "Model file size:",
        f"{file_size:,}",
        "bytes"
    )

    if file_size > 0:

        print(
            "\n✓ CNN MODEL SAVED SUCCESSFULLY!"
        )

    else:

        print(
            "\n✗ ERROR: Model is still 0 bytes!"
        )

else:

    print(
        "\n✗ ERROR: Model file was not created!"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nNumber of classes:",
    NUM_CLASSES
)

print(
    f"\nValidation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    "\nAdditional file:"
)

print(
    "✓ training_history.pkl"
)

print(
    "\nReady for evaluation."
)