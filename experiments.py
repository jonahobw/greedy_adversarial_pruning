from nets import get_hyperparameters
import json
import utils
import torchvision
import ssl
from shrinkbench.experiment import TrainingExperiment, PruningExperiment
from nets import best_model
from pathlib import Path
import os
import logging

logger = logging.getLogger('Experiment')


class Experiment:
    def __init__(
        self,
        experiment_number,
        dataset,
        model_type,
        model_path=None,
        best_model_metric="val_acc1",
        quantization=None,
        prune_method=None,
        prune_compression=None,
        prune_kwargs=None,
        finetune_epochs=None,
        attack_method=None,
        attack_kwargs=None,
        email=None,
        gpu=None,
        debug = False,
        kwargs=None,
    ):
        """

        :param experiment_number:
        :param dataset:
        :param model_type:
        :param model_path: optionally provide a path to the model
        :param quantization: the modulus for quantization
        :param prune_method:
        :param prune_compression:
        :param prune_kwargs:
        :param finetune_epochs: number of training epochs after pruning/quantization
        :param attack_method:
        :param attack_kwargs:
        :param gpu:
        :param debug
        :param kwargs:
        """

        self.paths, self.name = self.check_folder_structure(
            experiment_number,
            dataset,
            model_type,
            quantization,
            prune_method,
            attack_method,
            finetune_epochs,
        )
        self.experiment_number = experiment_number
        self.dataset = dataset
        self.model_type = model_type
        self.model_path = model_path
        self.train_from_scratch =  True if not self.model_path else False
        self.train_kwargs, self.prune_kwargs = get_hyperparameters(model_type)
        self.best_model_metric = best_model_metric
        self.quantization = quantization
        self.prune_method = prune_method
        self.prune_compression = prune_compression
        self.finetune_epochs = finetune_epochs
        self.attack_method = attack_method
        self.attack_kwargs = attack_kwargs
        self.gpu = gpu
        if self.gpu is not None:
            if self.train_kwargs is not None:
                self.train_kwargs["gpu"] = gpu
            if self.prune_kwargs is not None and "train_kwargs" in self.prune_kwargs:
                self.prune_kwargs["gpu"] = gpu
        if debug:
            self.train_kwargs['train_kwargs']['epochs'] = 1
        self.email = email

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.params = self.save_variables()

    def save_variables(self):
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
            "attack_method": self.attack_method
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        return params

    def run(self):
        self.email(f"Experiment started for {self.name}", json.dumps(self.params))

        if self.train_from_scratch:
            self.train()
        if self.prune_method:
            self.prune()
        if self.quantization:
            self.quantize()
        self.attack()

        self.email(f"Experiment ended for {self.name}", json.dumps(self.params))

    def train(self):
        # trains a CNN from scratch

        assert self.model_path == None, (
            "Training is done from scratch, should not be providing a pretrained model"
            "to train again."
        )
        self.train_exp = TrainingExperiment(
            dataset=self.dataset,
            model=self.model_type,
            path=self.paths["model"],
            checkpoint_metric=self.best_model_metric,
            **self.train_kwargs,
        )
        self.train_exp.run()

        self.model_path = best_model(self.train_exp.path, metric=self.best_model_metric)
        self.email(f"Training for {self.name} completed.")

    def prune(self):
        """
        Prunes a CNN.

        Can either prune an existing CNN from a different experiment (when model_path
        is provided as a class parameter) or can prune a model that was also trained during
        this experiment (model_path not provided).
        :return:
        """
        assert self.model_path != None, (
            "Either a path to a trained model must be provided or training parameters must be "
            "provided to train a model from scratch."
        )

        prune_args = self.train_kwargs
        prune_args['train_kwargs']['epochs'] = self.finetune_epochs
        if self.prune_kwargs is None:
            self.prune_kwargs = prune_args
        else:
            self.prune_kwargs.update(prune_args)

        self.prune_exp = PruningExperiment(
            dataset=self.dataset,
            model=self.model_type,
            strategy=self.prune_method,
            compression=self.prune_compression,
            resume=self.model_path,
            path=self.paths["model"],
            **self.prune_kwargs,
        )
        self.prune_exp.run()
        self.email(f"Pruning for {self.name} completed.")

    def quantize(self):
        pass

    def attack(self):
        pass

    def check_folder_structure(
        self,
        experiment_number,
        dataset,
        model_type,
        quantize,
        prune,
        attack,
        finetune_epochs,
    ):
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

        if quantize:
            path_dict["model"] = (
                path_dict["model_dataset"]
                / f"{model_type.lower()}_{quantize}_quantization"
            )
        elif prune:
            model_folder_name = f"{model_type.lower()}_{prune}"
            assert finetune_epochs > 0, "Number of finetuning epochs must be provided when pruning"
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

        for folder_name in path_dict:
            if not path_dict[folder_name].exists():
                path_dict[folder_name].mkdir(parents=True, exist_ok=True)

        # set environment variable to be used by shrinkbench
        os.environ["DATAPATH"] = str(path_dict["datasets"])

        return path_dict, model_folder_name


if __name__ == "__main__":
    for model_type in ["VGG", "GoogLeNet", "MobileNetV2", "ResNet18"]:
        a = Experiment(0, "CIFAR10", model_type, prune="full")
