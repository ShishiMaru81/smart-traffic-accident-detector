import os
import shutil
import random

random.seed(42)

source = "data"

classes = ["Accident", "NonAccident"]

for cls in classes:

    
    src_folder = os.path.join(source, cls, cls)
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    random.shuffle(files)

    total = len(files)

    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for split, file_list in zip(
        ["Train","Check","Test"],
        [train_files,val_files,test_files]
    ):

        dest_folder = os.path.join(source, split, cls)
        os.makedirs(dest_folder, exist_ok=True)

        for file in file_list:
            shutil.copy(
                os.path.join(src_folder,file),
                os.path.join(dest_folder,file)
            )

    print(cls,"done")