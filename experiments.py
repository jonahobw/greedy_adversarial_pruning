"""
Run experiments that train, prune and finetune or quantize, and attack a DNN.
"""

# pylint: disable=import-error, too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-locals, logging-fstring-interpolation, invalid-name
# pylint: disable=unspecified-encoding

import json
from pathlib import Path
import os
import logging
from typing import Callable
from shrinkbench.experiment import TrainingExperiment, PruningExperiment
from nets import get_hyperparameters, best_model

logger = logging.getLogger("Experiment")


class Experiment:
    """
    Creates a DNN, trains it on CIFAR10, then prunes and finetunes or quantizes, and attacks it.
    """

    def __init__(
        self,
        experiment_number: int,
        dataset: str,
        model_type: str,
        model_path: str = None,
        best_model_metric: str = "val_acc1",
        quantization: int = None,
        prune_method: str = None,
        prune_compression: int = None,
        finetune_epochs: int = None,
        attack_method: str = None,
        attack_kwargs: dict = None,
        email: Callable = None,
        gpu: int = None,
        debug: int = False,
        kwargs: dict = None,
    ):

        """
        Initialize all class variables.

        :param experiment_number: the overall number of the experiment.
        :param dataset: dataset to train on.
        :param model_type: architecture of model.
        :param model_path: optionally provide a path to a pretrained model.  If specified,
            the training will be skipped as the model is already trained.
        :param best_model_metric: metric to use to determine which is the best model.
        :param quantization: the modulus for quantization.
        :param prune_method: the strategy for the pruning algorithm.
        :param prune_compression: the desired ratio of parameters in original to pruned model.
        :param finetune_epochs: number of training epochs after pruning/quantization.
        :param attack_method: the method for generating adversarial inputs.
        :param attack_kwargs: arguments to be passed to the attack function.
        :param email: callback function which has a signature of (subject, message).
        :param gpu: the number of the gpu to run on.
        :param debug: whether or not to run in debug mode.  If specified, then
            should be an integer representing how many training/finetuning epochs to run.
            Also this will only run one batch for training/validation/fine-tuning, so the
            experimental results with this option specified are not valid.
        :param kwargs: additional keyword arguments (currently none).
        """

        self.paths, self.name = check_folder_structure(
            experiment_number,
            dataset,
            model_type,
            quantization,
            prune_method,
            attack_method,
            finetune_epochs,
            prune_compression,
        )
        self.experiment_number = experiment_number
        self.dataset = dataset
        self.model_type = model_type
        self.model_path = model_path

        # a model path may be provided to a pretrained model
        self.train_from_scratch = bool(self.model_path)

        # the hyperparameters for training/fine-tuning and pruning are dependent on the
        # model architecture
        self.train_kwargs, self.prune_kwargs = get_hyperparameters(
            model_type, debug=debug
        )
        self.best_model_metric = best_model_metric
        self.quantization = quantization
        self.prune_method = prune_method
        self.prune_compression = prune_compression
        self.finetune_epochs = finetune_epochs
        self.attack_method = attack_method
        self.attack_kwargs = attack_kwargs

        # there are 2 GPUs (may be changed for different hardware configurations)
        self.gpu = gpu
        if self.gpu in [0, 1]:
            self.train_kwargs["gpu"] = gpu
            self.prune_kwargs["gpu"] = gpu

        self.debug = debug
        if debug:
            logger.warning(
                f"Debug is {debug}.  Results will not be valid with this setting on."
            )
        self.email = email

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.params = self.save_variables()
        self.train_exp = None
        self.prune_exp = None

    def save_variables(self) -> None:
        """Save experiment parameters to a file in this experiment's folder"""

        params_path = self.paths["model"] / "experiment_params.json"
        params = {
            "experiment_number": self.experiment_number,
            "dataset": self.dataset,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "best_model_metric": self.best_model_metric,
            "quantization": self.quantization,
            "prune_method": self.prune_method,
            "prune_compression": self.prune_compression,
            "finetune_epochs": self.finetune_epochs,
            "attack_method": self.attack_method,
        }
        with open(params_path, "w") as file:
            json.dump(params, file, indent=4)
        return params

    def run(self) -> None:
        """Train, prune and finetune or quantize, and attack a DNN."""

        self.email(f"Experiment started for {self.name}", json.dumps(self.params))

        if self.train_from_scratch:
            self.train()
        if self.prune_method:
            self.prune()
        if self.quantization:
            self.quantize()
        self.attack()

        self.email(f"Experiment ended for {self.name}", json.dumps(self.params))

    def train(self) -> None:
        """Train a CNN from scratch."""

        assert self.model_path is None, (
            "Training is done from scratch, should not be providing a pretrained model"
            "to train again."
        )
        self.train_exp = TrainingExperiment(
            dataset=self.dataset,
            model=self.model_type,
            path=self.paths["model"],
            checkpoint_metric=self.best_model_metric,
            debug=self.debug,
            **self.train_kwargs,
        )
        self.train_exp.run()

        # set the model path to be the path to the best model from training.
        self.model_path = best_model(self.train_exp.path, metric=self.best_model_metric)
        self.email(f"Training for {self.name} completed.")

    def prune(self) -> None:
        """
        Prunes a CNN.

        Can either prune an existing CNN from a different experiment (when model_path
        is provided as a class parameter) or can prune a model that was also trained during
        this experiment (model_path not provided).
        :return: None
        """

        assert self.model_path is not None, (
            "Either a path to a trained model must be provided or training parameters must be "
            "provided to train a model from scratch."
        )
        logger.info(f"Pruning the model saved at {self.model_path}")

        self.prune_kwargs["train_kwargs"]["epochs"] = self.finetune_epochs

        self.prune_exp = PruningExperiment(
            dataset=self.dataset,
            model=self.model_type,
            strategy=self.prune_method,
            compression=self.prune_compression,
            checkpoint_metric=self.best_model_metric,
            resume=str(self.model_path),
            path=str(self.paths["model"]),
            debug=self.debug,
            **self.prune_kwargs,
        )
        self.prune_exp.run()
        self.email(f"Pruning for {self.name} completed.")

    def quantize(self):
        """Quantize a DNN."""

    def attack(self):
        """Attack a DNN."""


def check_folder_structure(
    experiment_number: int,
    dataset: str,
    model_type: str,
    quantize: int,
    prune: str,
    attack: str,
    finetune_epochs: int,
    compression: int,
) -> (str, dict):
    """
    Setup the paths to the dataset and experiment folders to follow the folder schema.
    For the folder schema, an experiment is defined on 2 levels: on one level, an
    experiment is defined by a model architecture, and pruning method, # of fine-tuning
    iterations, and compression ratio or quanization modulus, and experiments can be run
    for all combinations of the enumerated parameters.  On a high level, an experiment is
    defined as 1 run of all the experiments on the lower level.  Having multiple of the
    higher experiments means having duplicate runs of all model architecture,
    pruning method, # of fine-tuning iterations, and compression ratio or quanization
    modulus combinations.

    The folder schema is described below.

    aicas/
        datasets/
            CIFAR10/
        experiments/
            # high level experiment folders
            experiment1/
                VGG/
                    cifar10/
                    # low level experiment folders
                        vgg_<pruning_method>_<compression>_<finetuning_iterations>/
                            shrinkbench folder generated for training/
                            shrinkbench folder generated for pruning/
                            attacks/
                                <attack_method>/
                                    images/
                                    attack_results.csv
                            experiment_params.json
                        ...
                ResNet20/
                GoogLeNet/
                MobileNet/
            experiment2/
                ...

    :param experiment_number: the overall number of the experiment.
    :param dataset: dataset to train on.
    :param model_type: architecture of model.
    :param quantize: the modulus for quantization.
    :param prune_method: the strategy for the pruning algorithm.
    :param attack: the method for generating adversarial inputs.
    :param finetune_epochs: number of training epochs after pruning/quantization.
    :param compression: the desired ratio of parameters in original to pruned model.

    :return: a tuple (dict, str).  The string is the name of the model folder.
        The dictionary has keys which are the names of paths and the values are
        pathlib.Path object.  The keys are:
        'root': the root directory of the repository
        'datasets': path to the datasets folder
        'experiments': path to the folder of all experiments.
        'experiment': path to the folder of the experiment of which this experiment
            is a part of.
        'dataset': path to the folder storing the dataset that the DNN will be trained on.
        'model_type': path to the subfolder of the experiment folder for this model architecture
        'model_dataset' path to the subfolder of the 'model_type' folder for this dataset
        'model': the folder in which all data of this experiment will be saved.
    """

    root = Path.cwd()

    path_dict = {
        "root": root,
        "datasets": root / "datasets",
        "experiments": root / "experiments",
        "experiment": root / "experiments" / f"experiment_{experiment_number}",
    }
    path_dict["dataset"] = path_dict["datasets"] / dataset
    path_dict["model_type"] = path_dict["experiment"] / model_type
    path_dict["model_dataset"] = path_dict["model_type"] / dataset

    # model folder name must be unique for each variation of
    #   pruning/quantizing/finetuning for a certain model architecture
    model_folder_name = f"{model_type.lower()}"

    if quantize:
        path_dict["model"] = (
            path_dict["model_dataset"] / f"{model_type.lower()}_{quantize}_quantization"
        )
    elif prune:
        model_folder_name += f"_{prune}_{compression}_compression"
        assert (
            finetune_epochs > 0
        ), "Number of finetuning epochs must be provided when pruning"
    if finetune_epochs:
        model_folder_name += f"_{finetune_epochs}_finetune_iterations"
        path_dict["model"] = path_dict["model_dataset"] / model_folder_name
    else:
        path_dict["model"] = path_dict["model_dataset"] / model_type.lower()

    # see if experiment has been run already:
    if path_dict["model"].exists():
        logger.warning(f"Path {path_dict['model']} already exists.")

    if attack and "model" in path_dict:
        path_dict["attacks"] = path_dict["model"] / "attacks"
        path_dict["attack"] = path_dict["attacks"] / attack

    for folder_name in path_dict.items():
        if not path_dict[folder_name].exists():
            path_dict[folder_name].mkdir(parents=True, exist_ok=True)

    # set environment variable to be used by shrinkbench
    os.environ["DATAPATH"] = str(path_dict["datasets"])

    return path_dict, model_folder_name


if __name__ == "__main__":
    for architecture in ["VGG", "GoogLeNet", "MobileNetV2", "ResNet18"]:
        a = Experiment(0, "CIFAR10", architecture, prune_method="full")
