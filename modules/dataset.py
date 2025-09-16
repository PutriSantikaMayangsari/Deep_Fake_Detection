# modules/dataset.py

import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, sequences, labels, transform):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        transformed_sequence = torch.stack([self.transform(img) for img in sequence])
        return transformed_sequence, torch.tensor(label, dtype=torch.long)