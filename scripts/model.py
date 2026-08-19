"""A deliberately simple 1D-CNN baseline for 12-lead ECG classification.

Kept small on purpose: this is a baseline to get the pipeline working end to
end and to have a credible, explainable starting point. Swapping in a
resnet-1d or transformer later is a natural "next step" to mention in an
interview, not something to over-engineer on day one.
"""
import torch
import torch.nn as nn


class ECGConvNet(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool -> works for any input length
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, leads, time)
        x = self.features(x)
        return self.classifier(x)
