import os
import shutil
import zipfile

import requests
import matplotlib.pyplot as plt
from torchvision import datasets
from tqdm.auto import tqdm


def download_and_preview(dataset_name, dataset_folder="data", num_images_per_split=5, keep_zip=False, preview=False):
    print(f"Dataset '{dataset_name}'")
    url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip"
    download_path = os.path.join(dataset_folder, f"{dataset_name}.zip")
    target_folder = os.path.join(dataset_folder, dataset_name)

    os.makedirs(dataset_folder, exist_ok=True)

    print("Loading zip file...", end="", flush=True)
    if not os.path.isfile(download_path) or os.path.exists(target_folder):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            with open(download_path, "wb") as f, tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"{dataset_name}.zip",
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(" --> done!")

    print("Unziping...", end="", flush=True)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    with zipfile.ZipFile(download_path, "r") as zf:
        zf.extractall(dataset_folder)
    print(" --> done!")

    if not keep_zip and os.path.isfile(download_path):
        os.remove(download_path)

    print(f"Provided splits: {os.listdir(target_folder)}")
    dataset = datasets.ImageFolder(target_folder)

    inds_to_show = {i: [] for i, _ in enumerate(dataset.classes)}
    classes_full = 0
    for dataset_ind in range(len(dataset)):
        _, split_ind = dataset[dataset_ind]
        if len(inds_to_show[split_ind]) == num_images_per_split:
            continue
        inds_to_show[split_ind].append(dataset_ind)
        if len(inds_to_show[split_ind]) == num_images_per_split:
            classes_full += 1
        if classes_full == len(dataset.classes):
            break

    for split_name in sorted(dataset.classes):
        split_ind = dataset.class_to_idx[split_name]
        print(f"Split '{split_name}' of dataset '{dataset_name}'", end="")
        split_folder = os.path.join(target_folder, split_name)
        print(f" --> size: {len(os.listdir(split_folder))}")

        if preview:
            plt.subplots(1, num_images_per_split, figsize=(5 * num_images_per_split, 5))
            plt.suptitle(f"{dataset_name} ~ {split_name}", y=0.95)
            for i, dataset_ind in enumerate(inds_to_show[split_ind]):
                plt.subplot(1, num_images_per_split, i + 1)
                plt.imshow(dataset[dataset_ind][0])
                plt.xticks([])
                plt.yticks([])
            plt.show()

    print("\n----------------------------\n")
    return target_folder


if __name__ == "__main__":
    download_and_preview("ukiyoe2photo")
