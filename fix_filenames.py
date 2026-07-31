import os

folders = [
    "dataset_split/train/live",
    "dataset_split/train/spoof",
    "dataset_split/val/live",
    "dataset_split/val/spoof",
    "dataset_split/test/live",
    "dataset_split/test/spoof",
]

for folder in folders:

    for filename in os.listdir(folder):

        new_name = filename.strip().replace("\\", "").replace("/", "")

        if filename != new_name:

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)

            print(f"Renamed: {filename} -> {new_name}")

print("\nDone!")