"""
Script for running adversarial attacks on trained models as part of the model evaluation pipeline.

This module enables evaluation of model robustness to adversarial examples, supporting attacks such as PGD.
"""

from pathlib import Path
from shrinkbench.models import resnet20
from shrinkbench.experiment import AttackExperiment
import numpy as np
import os

model_path = (
    r"C:\Users\Jonah\Desktop\Jonah\0Grad_1\Research\code\aicas\experiments\experiment_0"
    r"\googlenet\CIFAR10\googlenet_GlobalMagGrad_2_compression_40_finetune_iterations"
    r"\20211101-111936-I0Y6-8bd05137846f1a27442e8665fcd4d428\checkpoints\checkpoint-232.pt"
)

dl_kwargs = {"batch_size": 2, "pin_memory": False, "num_workers": 1}
attack_params = {"eps": 2 / 255, "eps_iter": 0.001, "nb_iter": 5, "norm": np.inf}
path = Path.cwd()
os.environ["DATAPATH"] = str(path / "datasets")

a = AttackExperiment(
    model_path=model_path,
    model_type=resnet20(),
    dataset="CIFAR10",
    dl_kwargs=dl_kwargs,
    train=True,
    attack="PGD",
    attack_params=attack_params,
    path=path,
)
a.run()
