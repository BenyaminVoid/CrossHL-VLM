from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, Subset


DATASET_FOLDERS = {
    "Trento": "Trento11x11",
    "Houston": "HoustonDataset",
    "MUUFL": "MUUFL_Dataset",
}


class HSILidarDataset(Dataset):
    def __init__(self, root, dataset="Trento", split="train"):
        if dataset not in DATASET_FOLDERS:
            raise ValueError(f"Unknown dataset '{dataset}'.")
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        folder = Path(root) / DATASET_FOLDERS[dataset]
        hsi_name = "HSI_Tr.mat" if split == "train" else "HSI_Te.mat"
        lidar_name = "LIDAR_Tr.mat" if split == "train" else "LIDAR_Te.mat"
        label_name = "TrLabel.mat" if split == "train" else "TeLabel.mat"

        hsi = loadmat(folder / hsi_name)["Data"].astype(np.float32)
        lidar = loadmat(folder / lidar_name)["Data"].astype(np.float32)
        labels = loadmat(folder / label_name)["Data"]

        self.hs_image = torch.from_numpy(hsi).to(torch.float32).permute(0, 3, 1, 2)
        self.lidar_image = torch.from_numpy(lidar).to(torch.float32).permute(0, 3, 1, 2)
        self.lbls = (torch.from_numpy(labels) - 1).long().reshape(-1)

    def __len__(self):
        return self.hs_image.shape[0]

    def __getitem__(self, index):
        return self.hs_image[index], self.lidar_image[index], self.lbls[index]


def make_fewshot_indices(labels, pct, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = labels.detach().cpu()
    all_indices = torch.arange(len(labels))
    selected = []
    per_class_counts = {}

    for class_id in torch.unique(labels).tolist():
        class_indices = all_indices[labels == class_id]
        n_total = len(class_indices)
        n_keep = max(1, int(round(float(pct) * n_total)))
        perm = class_indices[torch.randperm(n_total, generator=generator)]
        keep = perm[:n_keep]
        selected.append(keep)
        per_class_counts[int(class_id)] = int(n_keep)

    selected = torch.cat(selected).tolist()
    return selected, per_class_counts


def make_fewshot_subset(dataset, pct, seed):
    selected, per_class_counts = make_fewshot_indices(dataset.lbls, pct, seed)
    return Subset(dataset, selected), per_class_counts


def make_kshot_indices(labels, shots, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = labels.detach().cpu()
    all_indices = torch.arange(len(labels))
    selected = []
    per_class_counts = {}

    for class_id in torch.unique(labels).tolist():
        class_indices = all_indices[labels == class_id]
        n_total = len(class_indices)
        n_keep = min(int(shots), n_total)
        perm = class_indices[torch.randperm(n_total, generator=generator)]
        keep = perm[:n_keep]
        selected.append(keep)
        per_class_counts[int(class_id)] = int(n_keep)

    selected = torch.cat(selected).tolist()
    return selected, per_class_counts


def make_kshot_subset(dataset, shots, seed):
    selected, per_class_counts = make_kshot_indices(dataset.lbls, shots, seed)
    return Subset(dataset, selected), per_class_counts


def apply_spectral_perturbation(hsi_batch, noise_std=0.0, gain_std=0.0):
    if noise_std <= 0 and gain_std <= 0:
        return hsi_batch

    out = hsi_batch
    if gain_std > 0:
        bands = out.shape[1]
        gain = 1.0 + torch.randn(
            out.shape[0],
            bands,
            1,
            1,
            device=out.device,
            dtype=out.dtype,
        ) * gain_std
        out = out * gain

    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std

    return out
