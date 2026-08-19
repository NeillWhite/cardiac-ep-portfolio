"""PyTorch Dataset wrapping preprocessed PTB-XL arrays."""
import numpy as np
import torch
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    """
    Expects X of shape (N, time, leads) and y of shape (N,) as produced by
    scripts/preprocess.py. Normalizes each lead to zero mean / unit variance
    per-record, and returns tensors shaped (leads, time) for Conv1d.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        signal = self.X[idx]  # (time, leads)
        # per-lead z-score normalization, guarding against constant/flat leads
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std

        signal = torch.from_numpy(signal.T).float()  # (leads, time)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return signal, label
