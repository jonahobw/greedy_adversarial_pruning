from torchvision import datasets
from pathlib import Path

download_path = Path.cwd() / 'datasets' / 'CIFAR10'

if not download_path.exists():
    download_path.mkdir(parents=True, exist_ok=True)

cifar10 = datasets.CIFAR10(root=download_path, download=True)
