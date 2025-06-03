import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from sparseml.pytorch.torchvision.train import evaluate


class DummyDataset(Dataset):
    def __init__(self, length=5):
        self.length = length
        self.data = [torch.zeros(1, 2, 2) for _ in range(length)]
        self.labels = [0 for _ in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx % self.length], self.labels[idx % self.length]


class PaddedSampler(Sampler):
    def __iter__(self):
        return iter([0, 1, 2, 3, 4, 0])

    def __len__(self):
        return 6


class ConstantModel(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        out = torch.zeros(batch, 5)
        out[:, 0] = 10.0
        return out


def test_evaluate_handles_padding():
    dataset = DummyDataset()
    sampler = PaddedSampler()
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    model = ConstantModel()
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, criterion, loader, device="cpu", print_freq=100)

    assert metrics.acc1.count == len(dataset)
    assert metrics.acc1.global_avg == pytest.approx(100.0)
