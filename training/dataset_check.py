import os

DATASET_PATH = "C:/Users/VANSHIKA/Desktop/Python/Python313/internship_projects/TrafficSignClassification/dataset/TrafficSignDataset/traffic_Data/DATA"

print("=" * 60)
print("TRAFFIC SIGN DATASET INFORMATION")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found.")
    exit()

class_folders = [
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
]

print(f"\nTotal Classes: {len(class_folders)}")

total_images = 0

print("\nImages per class:")
print("-" * 40)

for class_name in sorted(class_folders):
    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".ppm")
        )
    ]

    print(f"Class {class_name}: {len(images)} images")

    total_images += len(images)

print("-" * 40)
print(f"Total Images: {total_images}")