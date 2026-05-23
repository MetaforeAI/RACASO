"""CIFAR-10 loader using torchvision.datasets.

Downloads to ``BENCH_DATA_DIR`` (or ``bench/datasets/_cache``) on first
call. Returns ``(train_loader, test_loader)``. Standard CIFAR-10
augmentation: random crop with 4-pixel padding + horizontal flip on
train, none on test. Normalization uses the canonical per-channel
means/stds.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from bench.datasets import get_data_root


_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10.

    Default ``num_workers=0`` runs data loading in-process — slightly
    slower than worker pools but avoids fork/spawn issues on containers
    (RunPod, Docker-without-IPC, etc.) where ConnectionResetError on
    worker handles is common.
    """
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for CIFAR-10. Install: pip install torchvision"
        ) from exc

    root = str(get_data_root())

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf
    )

    pin_memory = (device == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
