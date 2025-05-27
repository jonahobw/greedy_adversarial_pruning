"""
Main experiment orchestration.

Defines the Experiment class to manage training, pruning (including GAP), quantization,
and adversarial evaluation of deep neural networks as described in the GAP paper.
"""

# pylint: disable=import-error, too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-locals, logging-fstring-interpolation, invalid-name
# pylint: disable=unspecified-encoding
import copy
import json
import csv
from pathlib import Path
import os
import logging
from typing import Callable
import datetime

import numpy as np
from shrinkbench.experiment import (
    TrainingExperiment,
    PruningExperiment,
    AttackExperiment,
    QuantizeExperiment,
)
from nets import get_hyperparameters, best_model
from evaulate_model import Model_Evaluator
from experiment_utils import format_path, find_recent_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Experiment")


class Experiment:
    """
    Creates a DNN, trains it on CIFAR10, then prunes and finetunes or quantizes, and attacks it.
    Orchestrates the full experimental pipeline: training, pruning, quantization, and adversarial evaluation.
    Handles experiment folder structure, parameter management, and result aggregation.
    """
    def __init__(
        self,
        experiment_number: int = None,
        dataset: str = None,
        model_type: str = None,
        model_path: str = None,
        resume: bool = False,
        best_model_metric: str = None,
        quantization: bool = False,
        prune_method: str = None,
        prune_compression: int = 1,
        finetune_epochs: int = None,
        attack_method: str = None,
        attack_kwargs: dict = None,
        skip_attack: bool = False,
        only_attack: bool = False,
        transfer_attack_model: str = None,
        email: Callable = None,
        email_verbose: bool = False,
        gpu: int = None,
        debug: int = False,
        save_one_checkpoint: bool = False,
        seed: int = None,
        train_kwargs: {} = None,
    ):
        """
        Initialize all class variables and set up experiment configuration.

        Args:
            experiment_number (int): The overall number of the experiment.
            dataset (str): Dataset to train on.
            model_type (str): Architecture of model.
            model_path (str, optional): Path to a pretrained model. If specified, training is skipped.
            resume (bool): Whether to continue an existing experiment.
            best_model_metric (str, optional): Metric to use for best model selection, default is to 
              use model at end of training.
            quantization (bool): Whether to quantize the model to int8after training/pruning.
            prune_method (str, optional): Pruning strategy.
            prune_compression (int): Desired compression ratio (original:pruned).
            finetune_epochs (int, optional): Number of epochs for fine-tuning after pruning.
            attack_method (str, optional): Adversarial attack method.
            attack_kwargs (dict, optional): Parameters for the attack.
            skip_attack (bool): If True, skip adversarial evaluation.
            only_attack (bool): If True, only run attack and transfer attack. This is only valid if a 
              model path is provided, and resume, is set to True, and transfer_attack_model is provided.
            transfer_attack_model (str, optional): Path to a model from which to construct a transfer attack by
              generating adversarial inputs on this model and testing them on the model associated with
              this experiment.  Only valid when attack_kwargs/attack_method are provided.
            email (Callable, optional): Callback for email notifications, which has a signature of (subject, message).
            email_verbose (bool): If True, sends an email at start and end of whole experiment, and end of
              training, pruning, fine-tuning, quantization, and attack.  If false, then only sends an
              email at the start and end of each experiment.
            email_verbose (bool): If True, sends an email at start and end of whole experiment, and end of
            training, pruning, fine-tuning, quantization, and attack.  If false, then only sends an
            email at the start and end of each experiment.
            gpu (int, optional): GPU index to use.
            debug (int, optional): If set, trains with only 1 epoch debug and specifies number of minibatches.
            save_one_checkpoint (bool): If True, only save one checkpoint per phase.
            seed (int, optional): Random seed for reproducibility.
            train_kwargs (dict, optional): Additional training parameters.
        """
        self.experiment_number = experiment_number
        self.dataset = dataset
        self.model_type = model_type
        self.model_path = format_path(model_path, directory="experiments")
        self.resume = resume
        self.best_model_metric = best_model_metric
        self.quantization = quantization
        self.prune_method = prune_method
        self.prune_compression = prune_compression
        self.finetune_epochs = finetune_epochs
        self.attack_method = attack_method
        self.attack_kwargs = attack_kwargs
        self.skip_attack = skip_attack
        self.only_attack = only_attack
        if self.only_attack:
            assert resume, "Resume must be set to True when only_attack is set to True."
            assert (
                model_path is not None
            ), "Model path must be provided when only_attack is set to True."
            assert (
                transfer_attack_model is not None
            ), "Transfer model must be provided when only_attack is set to True."
        self.save_one_checkpoint = save_one_checkpoint
        self.seed = seed
        self.transfer_attack_model = format_path(transfer_attack_model, directory="experiments")

        # Set up experiment folder structure and name
        self.paths, self.name = self.generate_paths()

        # Determine if training is from scratch or from a pretrained model
        self.train_from_scratch = not bool(self.model_path)

        # Get hyperparameters for training and pruning
        self.train_kwargs, self.prune_kwargs = get_hyperparameters(
            model_type, debug=debug
        )
        if train_kwargs is not None:
            self.train_kwargs["train_kwargs"].update(train_kwargs)

        self.gpu = gpu
        if self.gpu is not None:
            self.train_kwargs["gpu"] = gpu
            self.prune_kwargs["gpu"] = gpu

        self.debug = debug
        if debug:
            logger.warning(
                f"Debug is {debug}.  Results will not be valid with this setting on."
            )
        # Set up email callback
        self.email = email if email is not None else lambda x, y: 0
        self.email_verbose = email_verbose
        self.params = self.save_variables()
        self.train_exp = None
        self.prune_exp = None
        self.all_results = {}
        self.pre_quantized_model = None
        self.quantized = False

    def generate_paths(self):
        """
        Setup the paths to the experiment folders.

        Calls check_folder_structure() to generate paths.  If a model_path is provided, meaning that
        a model has already been trained, and self.resume is True, then will find the existing paths
        instead of creating them.

        Raises:
            ValueError: If a model path is provided, and any of the provided experiment_number, dataset,
            model_type do not match the ones found in the model path.  If self.resume is true, also raises
            this error if quantization, prune_method, attack_method, finetune_epochs, or prune_compression
            do not match the ones found using the model_path.
        Returns:
            tuple: (dict of paths, model folder name)
        """
        if self.model_path:
            m_path = str(self.model_path)
            sep = "\\" if "\\" in m_path else "/"

            if self.model_type:
                if self.model_type not in m_path:
                    raise ValueError(
                        f"Provided model type {self.model_type} but provided model path"
                        f"\n{self.model_path} does not include this model_type."
                    )
            else:
                self.model_type = m_path[
                    m_path.find(f"experiment_{self.experiment_number}") :
                ].split(sep)[1]

            if self.dataset:
                if self.dataset not in m_path:
                    raise ValueError(
                        f"Provided dataset {self.dataset} but provided model path"
                        f"\n{self.model_path} does not include this dataset."
                    )
            else:
                self.dataset = m_path[m_path.find(self.model_type) :].split(sep)[1]

            if self.resume:
                if self.experiment_number:
                    if f"experiment_{self.experiment_number}" not in m_path:
                        raise ValueError(
                            f"Provided experiment number {self.experiment_number} but provided model path"
                            f"\n{self.model_path} does not include this experiment number."
                        )
                else:
                    experiment_number = int(
                        m_path[m_path.find("experiment_") :].split("_")[1].split(sep)[0]
                    )

                already_pruned = False

                if self.prune_compression > 1:
                    # Only throw an error if there is a different compression already applied.
                    if "compression" in m_path:
                        if f"{self.prune_compression}_compression" not in m_path:
                            raise ValueError(
                                f"Provided pruning compression {self.prune_compression} but provided model path"
                                f"\n{self.model_path} does not include this prune compression."
                            )
                        else:
                            already_pruned = True
                else:
                    loc = m_path.find("compression")
                    if loc >= 0:
                        self.prune_compression = int(m_path[:loc].split("_")[-2])
                    else:
                        self.prune_compression = None

                if self.prune_method:
                    # Only throw an error if there is a different pruning already applied
                    #   (checked using already_pruned variable)
                    if self.prune_method not in m_path and already_pruned:
                        raise ValueError(
                            f"Provided pruning method {self.prune_method} but provided model path"
                            f"\n{self.model_path} does not include this pruning method."
                        )
                else:
                    if self.prune_compression:
                        self.prune_method = m_path[
                            : m_path.find(f"{self.prune_compression}_compression")
                        ].split("_")[-2]
                    else:
                        self.prune_method = None

                if self.finetune_epochs:
                    # Only throw an error if there is a different finetuning already applied
                    #   (checked using already_pruned variable)
                    if (
                        f"{self.finetune_epochs}_finetune_iterations" not in m_path
                        and already_pruned
                    ):
                        raise ValueError(
                            f"Provided finetune epochs {self.finetune_epochs} but provided model path"
                            f"\n{self.model_path} does not include this finetune_epochs."
                        )
                else:
                    if self.prune_method:
                        self.finetune_epochs = int(
                            m_path[: m_path.find("finetune")].split("_")[-2]
                        )
                    else:
                        self.finetune_epochs = None

        return check_folder_structure(
            experiment_number=self.experiment_number,
            dataset=self.dataset,
            model_type=self.model_type,
            quantize=self.quantization,
            prune=self.prune_method,
            attack=self.attack_method,
            finetune_epochs=self.finetune_epochs,
            compression=self.prune_compression,
        )

    def save_variables(self, write=True) -> None:
        """
        Save experiment parameters to a file in this experiment's folder.

        Args:
            write (bool): Whether to write the parameters to disk.
        Returns:
            str: JSON string of parameters.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        params_path = self.paths["model"] / f"experiment_params_{timestamp}.json"
        variables = [
            "name",
            "experiment_number",
            "dataset",
            "model_type",
            "model_path",
            "best_model_metric",
            "quantization",
            "prune_method",
            "prune_compression",
            "finetune_epochs",
            "attack_method",
            "attack_kwargs",
            "email_verbose",
            "gpu",
            "debug",
            "save_one_checkpoint",
            "seed",
            "train_from_scratch",
            "train_kwargs",
            "prune_kwargs",
            "dataset",
            "skip_attack",
            "only_attack",
            "transfer_attack_model",
        ]
        params = {
            x: str(getattr(self, x))
            if isinstance(getattr(self, x), Path)
            else getattr(self, x)
            for x in variables
        }
        default = lambda x: "json encode error"
        params = json.dumps(params, skipkeys=True, indent=4, default=default)
        if write:
            with open(params_path, "w") as file:
                file.write(params)
        return params

    def check_already_done(self):
        """
        Checks if the experiment has already been trained, pruned, or quantized.

        Returns:
            tuple: (already_trained, already_pruned, already_quantized)
        """
        subfolders = [x.name for x in self.paths["model"].iterdir() if x.is_dir()]
        return "train" in subfolders, "prune" in subfolders, "quantize" in subfolders

    def run(self) -> None:
        """
        Run the full experiment pipeline: training, pruning, quantization, and attacks.
        Sends email notifications at start and end.
        """
        self.email(f"Experiment started for {self.name}", self.params)
        (
            self.already_trained,
            self.already_pruned,
            self.already_quantized,
        ) = self.check_already_done()

        # Training phase
        if self.train_from_scratch:
            if not self.only_attack and not self.already_trained:
                self.train()
                self.evaluate(attack=False, subfolder="train")
            if self.attack_method and not self.skip_attack:
                self.attack(subfolder="train")
        # Pruning phase
        if self.prune_method:
            if not self.only_attack and not self.already_pruned:
                self.prune()
                self.evaluate(attack=False, subfolder="prune")
            if self.attack_method and not self.skip_attack:
                self.attack(subfolder="prune")
        # Quantization phase
        if self.quantization:
            if not self.only_attack and not self.already_quantized:
                self.quantize()
                self.evaluate(attack=False, subfolder="quantize")
            if self.attack_method and not self.skip_attack:
                self.attack(subfolder="quantize")

        results = self.save_results()
        self.email(f"Experiment ended for {self.name}", results)

    def train(self) -> None:
        """
        Train a CNN from scratch and save the best model checkpoint.
        """
        assert self.model_path is None, (
            "Training is done from scratch, should not be providing a pretrained model"
            "to train again."
        )
        self.train_exp = TrainingExperiment(
            dataset=self.dataset,
            model=self.model_type,
            path=str(self.paths["model"]),
            checkpoint_metric=self.best_model_metric,
            debug=self.debug,
            save_one_checkpoint=self.save_one_checkpoint,
            seed=self.seed,
            **self.train_kwargs,
        )
        self.train_exp.run()

        # Set the model path to the best model from training
        self.model_path = best_model(self.train_exp.path, metric=self.best_model_metric)

        if self.email_verbose:
            self.email(f"Training for {self.name} completed.", "")

    def prune(self) -> None:
        """
        Prune a CNN using the specified pruning method and fine-tune.
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
            save_one_checkpoint=self.save_one_checkpoint,
            seed=self.seed,
            attack_kwargs=self.attack_kwargs,
            **self.prune_kwargs,
        )
        self.prune_exp.run()

        # Set the model path to the best model from pruning
        self.model_path = best_model(self.prune_exp.path, metric=self.best_model_metric)

        if self.email_verbose:
            self.email(f"Pruning for {self.name} completed.", "")

    def quantize(self):
        """
        Quantize a DNN and update the model path to the quantized model.
        """
        self.pre_quantized_model = self.model_path

        quantize_exp = QuantizeExperiment(
            model_path=self.model_path,
            model_type=self.model_type,
            dataset=self.dataset,
            dl_kwargs=self.train_kwargs["dl_kwargs"],
            train=True,
            path=self.paths["model"],
            gpu=self.gpu,
            seed=self.seed,
            debug=self.debug,
        )

        quantize_exp.run()
        self.model_path = quantize_exp.save_model()
        self.quantized = True

        if self.email_verbose:
            self.email(f"Quantization for {self.name} completed.", "")

    def attack(self, subfolder):
        """
        Run an adversarial attack on the model and save results.

        Args:
            subfolder (str): The subfolder to save attack results to (e.g., 'train', 'prune', 'quantize').
        """
        save_folder = self.paths["model"] / subfolder

        default_pgd_args = {
            "eps": 2 / 255,
            "eps_iter": 0.001,
            "nb_iter": 10,
            "norm": np.inf,
        }

        train = self.attack_kwargs.pop("train")
        default_pgd_args.update(**self.attack_kwargs)
        logger.info(
            f"Attacking with parameters:\n{json.dumps(default_pgd_args, indent=4)}"
        )

        assert self.model_path is not None, (
            "Either a path to a trained model must be provided or training parameters must be "
            "provided to train a model from scratch."
        )

        pre_quantized_model = None
        if self.quantized:
            pre_quantized_model = self.pre_quantized_model

        attack_exp = AttackExperiment(
            model_path=self.model_path,
            model_type=self.model_type,
            dataset=self.dataset,
            dl_kwargs=self.train_kwargs["dl_kwargs"],
            train=train,
            attack=self.attack_method,
            attack_params=default_pgd_args,
            path=save_folder,
            transfer_model_path=str(self.transfer_attack_model)
            if self.transfer_attack_model
            else None,
            gpu=self.gpu,
            seed=self.seed,
            debug=self.debug,
            quantized=self.quantized,
            pre_quantized_model=pre_quantized_model,
        )

        attack_exp.run()
        results = attack_exp.results

        if subfolder not in self.all_results:
            self.all_results[subfolder] = {}
        if "attack" not in self.all_results[subfolder]:
            self.all_results[subfolder]["attack"] = {}
        self.all_results[subfolder]["attack"]["results"] = results

        if self.transfer_attack_model:
            attack_exp.run_transfer()
            transfer_results = attack_exp.transfer_results
            self.all_results[subfolder]["attack"]["transfer"] = transfer_results

        # Restore the 'train' key in attack_kwargs in case this function is called again
        self.attack_kwargs["train"] = train

        if self.email_verbose:
            self.email(
                f"{self.attack_method} attack on {self.model_type} Concluded.",
                json.dumps(self.attack_exp.save_variables(), indent=4),
            )

    def evaluate(self, attack, subfolder):
        """
        Evaluate the model using Model_Evaluator and store results.

        Args:
            attack (bool): Whether to run adversarial evaluation.
            subfolder (str): The subfolder to store results under.
        """
        a = Model_Evaluator(
            model_type=self.model_type,
            model_path=self.model_path,
            dataset=self.dataset,
            gpu=self.gpu,
            seed=self.seed,
            dl_kwargs=self.train_kwargs["dl_kwargs"],
            debug=self.debug,
            attack_method=self.attack_method,
            attack_kwargs=self.attack_kwargs,
            quantized=self.quantized,
            surrogate_model_path=self.pre_quantized_model,
        )
        model_eval_results = a.run(attack=attack)

        if subfolder not in self.all_results:
            self.all_results[subfolder] = {}
        self.all_results[subfolder].update(model_eval_results)

    def save_results(self):
        """
        Save all experiment results to a JSON file, merging with previous results if present.

        Returns:
            str: JSON string of all results.
        """
        all_info = {**self.all_results, **json.loads(self.params)}
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        params_path = self.paths["model"] / f"experiment_results_{timestamp}.json"

        previous_results_file = find_recent_file(
            folder=self.paths["model"], prefix="experiment_results_"
        )
        if isinstance(previous_results_file, Path):
            logger.info(
                f"Combining results with previous results file: \n{previous_results_file}"
            )
            with open(previous_results_file, "r") as file:
                previous_results = json.load(file)
            previous_results.update(all_info)
            all_info = previous_results

        default = lambda x: "json encode error"
        params = json.dumps(all_info, skipkeys=True, indent=4, default=default)
        with open(params_path, "w") as file:
            file.write(params)
        return params


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
                resnet20/
                    cifar10/
                    # low level experiment folders (model folders)
                        resnet20_<pruning_method>_<compression>_<finetuning_iterations>/
                            train/
                                attacks/
                                    <attack_method>/
                                        transfer_attacks/
                                        images/
                                        train_attack_results-*.json
                            prune/
                                attacks/
                            quantize/
                                attacks/
                            experiment_params.json
                        ...
                vgg_bn_drop/
                googlenet/
                mobilenet_v2/
                # csv to store high level results of all experiments
                logs.csv
            experiment2/
                ...

    Args:
        experiment_number (int): The overall number of the experiment.
        dataset (str): Dataset to train on.
        model_type (str): Architecture of model.
        quantize (int): The modulus for quantization.
        prune (str): The strategy for the pruning algorithm.
        attack (str): The method for generating adversarial inputs.
        finetune_epochs (int): Number of training epochs after pruning.
        compression (int): The desired ratio of parameters in original to pruned model.

    Returns:
        tuple: (dict of paths, model folder name)
        The string is the name of the model folder.
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

    if prune:
        assert (
            compression is not None and compression > 1
        ), "When pruning, must provide compression as a number > 1."
        model_folder_name += f"_{prune}_{compression}_compression"
        assert (
            isinstance(finetune_epochs, int) and finetune_epochs >= 0
        ), "Number of finetuning epochs must be provided when pruning"
        if finetune_epochs:
            model_folder_name += f"_{finetune_epochs}_finetune_iterations"
    # if quantize:
    #     model_folder_name += f"_quantization"

    path_dict["model"] = path_dict["model_dataset"] / model_folder_name

    # see if experiment has been run already:
    if path_dict["model"].exists():
        logger.warning(f"Path {path_dict['model']} already exists.")

    for folder_name in path_dict.keys():
        if not path_dict[folder_name].exists():
            path_dict[folder_name].mkdir(parents=True, exist_ok=True)

    path_dict["logs.csv"] = (
        path_dict["experiment"] / f"experiment_{experiment_number}_logs.csv"
    )

    # set environment variable to be used by shrinkbench
    os.environ["DATAPATH"] = str(path_dict["datasets"])

    return path_dict, model_folder_name


if __name__ == "__main__":
    for architecture in ["VGG", "GoogLeNet", "MobileNetV2", "ResNet18"]:
        a = Experiment(0, "CIFAR10", architecture, prune_method="full")
