"""
Greedy Adversarial Pruning (GAP) experimental framework.

This package provides tools for training, pruning, quantizing, and evaluating deep neural networks
with a focus on adversarial robustness and parameter efficiency, as described in the paper.
"""
from .experiment_utils import Email_Sender, format_path
from .experiments import Experiment
from .nets import get_train_hyperparameters, best_model
from .evaulate_model import Model_Evaluator
