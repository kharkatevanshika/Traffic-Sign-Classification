import os
import pickle
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# TRAFFIC SIGN DATA PREPROCESSING
# ============================================================

print("=" * 60)
print("TRAFFIC SIGN DATA PREPROCESSING")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_ROOT = os.path.join(
    BASE_DIR,
    "dataset",
    "TrafficSignDataset",
    "traffic_Data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("\nDataset path:")
print(DATASET_ROOT)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_ROOT):

    print("\nERROR: Dataset folder not found!")
    print(DATASET_ROOT)

    input("\nPress Enter to exit...")
    raise SystemExit


print("\nDataset folder found successfully!")


# ============================================================
# FIND NUMERIC CLASS FOLDERS RECURSIVELY
# ============================================================

print("\nSearching for class folders...")


class_folders = []

for root, dirs, files in os.walk(DATASET_ROOT):

    folder_name = os.path.basename(root)

    # Class folders should have numeric names
    if folder_name.isdigit():

        class_folders.append(
            (int(folder_name), root)
        )


# Remove duplicates
unique_classes = {}

for class_number, folder_path in class_folders:

    unique_classes[class_number] = folder_path


class_folders = sorted(
    unique_classes.items(),
    key=lambda x: x[0]
)


# ============================================================
# CHECK CLASSES
# ============================================================

if len(class_folders) == 0:

    print("\nERROR: No numeric class folders found!")

    print("\nPlease check your dataset structure.")

    input("\nPress Enter to exit...")
    raise SystemExit


print(
    f"\nFound {len(class_folders)} class folders."
)

print("-" * 60)


# ============================================================
# CREATE CLASS NAMES
# ============================================================

class_names = []

for class_number, folder_path in class_folders:

    class_names.append(
        str(class_number)
    )


print(
    "Total Classes:",
    len(class_names)
)


# ============================================================
# LOAD IMAGES
# ============================================================

images = []
labels = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".webp"
)


print("\nLoading images...")
print("-" * 60)


for new_label, (class_number, folder_path) in enumerate(
    class_folders
):

    image_count = 0

    for filename in os.listdir(folder_path):

        file_path = os.path.join(
            folder_path,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        try:

            # Open image
            image = Image.open(
                file_path
            ).convert("RGB")

            # Resize
            image = image.resize(
                (32, 32)
            )

            # Convert to numpy
            image_array = np.array(
                image,
                dtype=np.float32
            )

            images.append(
                image_array
            )

            labels.append(
                new_label
            )

            image_count += 1

        except Exception as e:

            print(
                f"Skipped {filename}: {e}"
            )

    print(
        f"Class {class_number}: "
        f"{image_count} images"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    images,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int64
)


print("\n" + "=" * 60)
print("DATA LOADED SUCCESSFULLY")
print("=" * 60)

print(
    "Images:",
    X.shape
)

print(
    "Labels:",
    y.shape
)


# ============================================================
# CHECK DATA
# ============================================================

if len(X) == 0:

    print("\nERROR: No images loaded!")
    raise SystemExit


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing images...")

X = X / 255.0

print(
    "Pixel values are now between 0 and 1."
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nSplitting dataset...")


# Try stratified split first
try:

    X_train, X_val, y_train, y_val = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

except ValueError:

    print(
        "\nWarning: Stratified split could not be used."
    )

    print(
        "Using normal random split."
    )

    X_train, X_val, y_train, y_val = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


print(
    "\nTraining images:",
    X_train.shape
)

print(
    "Validation images:",
    X_val.shape
)


# ============================================================
# SAVE DATA
# ============================================================

print("\nSaving processed data...")


np.save(
    os.path.join(
        MODEL_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        MODEL_DIR,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        MODEL_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        MODEL_DIR,
        "y_val.npy"
    ),
    y_val
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "class_names.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        class_names,
        f
    )


# ============================================================
# VERIFY FILES
# ============================================================

print("\nChecking generated files...")

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

    if os.path.exists(file_path):

        size = os.path.getsize(
            file_path
        )

        print(
            f"✓ {filename} "
            f"({size:,} bytes)"
        )

    else:

        print(
            f"✗ {filename} NOT CREATED"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nTotal classes:",
    len(class_names)
)

print(
    "Total images:",
    len(X)
)

print(
    "Training images:",
    len(X_train)
)

print(
    "Validation images:",
    len(X_val)
)

print(
    "\nReady for CNN training."
)