import copy
import json
import pathlib
import os

import numpy as np
from tqdm import tqdm
import torch
import torchvision.models
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from shrinkbench.experiment import QuantizeExperiment, DNNExperiment
from shrinkbench import models
from shrinkbench.models.head import mark_classifier
from shrinkbench.metrics import model_size, flops, accuracy, correct
from shrinkbench.util import OnlineStats
from experiment_utils import format_path

attacks = {"pgd": projected_gradient_descent}


class Model_Evaluator(DNNExperiment):
    default_dl_kwargs = {
        "batch_size": 128,
        "pin_memory": False,
        "num_workers": 4,
    }

    def __init__(
        self,
        model_type,
        model_path,
        dataset="CIFAR10",
        gpu=None,
        seed: int=None,
        dl_kwargs: {} = None,
        debug=None,
        attack_method="pgd",
        attack_kwargs=None,
        quantized=False,
        surrogate_model_path =None,
    ):
        super().__init__(seed=seed)
        if seed:
            self.fix_seed(seed, deterministic=True)

        # set environment variable to be used by shrinkbench
        os.environ["DATAPATH"] = str(format_path("datasets"))

        if dl_kwargs:
            self.dl_kwargs = dl_kwargs
        else:
            self.dl_kwargs = self.default_dl_kwargs

        self.path = format_path(model_path)
        self.resume = self.path

        self.gpu = gpu
        self.dataset = dataset
        self.debug = debug
        self.model_type = model_type
        self.model_path = model_path
        self.surrogate_model_path = surrogate_model_path

        self.build_dataloader(dataset=dataset, **self.dl_kwargs)
        self.quantized = quantized
        if not quantized:
            self.build_model(
                self.model_type,
                pretrained=False,
                resume=self.model_path,
                dataset=dataset,
            )
            self.surrogate_model = None
        else:
            self.surrogate_model = self.build_model(
                self.model_type,
                pretrained=False,
                resume=self.surrogate_model_path,
                dataset=dataset,
            )
            self.model = QuantizeExperiment.load_quantized_model(
                path=self.model_path,
                model_type=self.model_type,
                dataset=self.dataset,
            )
        self.attack_method = attack_method
        self.attack_kwargs = attack_kwargs

    def clean_acc(self):
        res = list(
            accuracy(
                model=self.model,
                dataloader=self.train_acc_dl,
                topk=(1, 5),
                debug=self.debug,
            )
        )
        self.clean_train_acc1 = res[0]
        self.clean_train_acc5 = res[1]
        res = list(
            accuracy(
                model=self.model, dataloader=self.val_dl, topk=(1, 5), debug=self.debug
            )
        )
        self.clean_val_acc1 = res[0]
        self.clean_val_acc5 = res[1]

    def prune_metrics(self):
        """Collect the pruning metrics."""
        # Model Size
        size, size_nz = model_size(self.model)
        self.model_size = size
        self.model_size_nz = size_nz
        self.compression_ratio = size / size_nz
        self.model_file_size = pathlib.Path(self.model_path).stat().st_size

        x, y = next(iter(self.val_dl))
        if not self.quantized:
            x, y = x.to(self.device), y.to(self.device)
        else:
            x, y = x.to('cpu'), y.to('cpu')

        # FLOPS
        ops, ops_nz = flops(self.model, x, quantized=self.quantized)
        self.flops = ops
        self.flops_nz = ops_nz
        self.theoretical_speedup = ops / ops_nz

    def adv_acc(self, train=True):
        assert self.attack_method is not None
        assert self.attack_kwargs is not None

        self.model.eval()

        if train:
            dl = self.train_acc_dl
            data = "train"
        else:
            dl = self.val_dl
            data = "test"

        results = {"adversarial_dataset": data, "inputs_tested": 0}

        clean_acc1 = OnlineStats()
        clean_acc5 = OnlineStats()
        adv_acc1 = OnlineStats()
        adv_acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{attack_name} on {data} dataset")

        for i, (x, y) in enumerate(epoch_iter, start=1):
            if self.debug is not None and i > self.debug:
                break
            x, y = x.to(self.device), y.to(self.device)
            if not self.quantized:
                x_adv = attacks[self.attack_method](self.model, x, **self.attack_kwargs)
            else:
                x_adv = attacks[self.attack_method](self.surrogate_model, x, **self.attack_kwargs)
                x, x_adv, y = x.to('cpu'), x_adv.to('cpu'), y.to('cpu')
            y_pred = self.model(x)  # model prediction on clean examples
            y_pred_adv = self.model(x_adv)  # model prediction on adversarial examples

            results["inputs_tested"] += y.size(0)

            clean_c1, clean_c5 = correct(y_pred, y, (1, 5))
            clean_acc1.add(clean_c1 / dl.batch_size)
            clean_acc5.add(clean_c5 / dl.batch_size)

            adv_c1, adv_c5 = correct(y_pred_adv, y, (1, 5))
            adv_acc1.add(adv_c1 / dl.batch_size)
            adv_acc5.add(adv_c5 / dl.batch_size)

            epoch_iter.set_postfix(
                clean_acc1=clean_acc1.mean,
                clean_acc5=clean_acc5.mean,
                adv_acc1=adv_acc1.mean,
                adv_acc5=adv_acc5.mean,
            )

        results["clean_acc1"] = clean_acc1.mean
        results["clean_acc5"] = clean_acc5.mean
        results["adv_acc1"] = adv_acc1.mean
        results["adv_acc5"] = adv_acc5.mean

        self.adv_results = results

    def print_results(self):
        metrics = {}
        for name in [
            "clean_train_acc1",
            "clean_train_acc5",
            "clean_val_acc1",
            "clean_val_acc5",
            "model_size",
            "model_size_nz",
            "compression_ratio",
            "flops",
            "flops_nz",
            "theoretical_speedup",
            "adv_results",
            "model_file_size",
        ]:
            if hasattr(self, name):
                metrics[name] = getattr(self, name)

        print(json.dumps(metrics, indent=4))
        return metrics

    def run(self, attack=False):
        print("Evaluating model ...\nGetting clean accuracy ...")
        self.clean_acc()
        print("Getting pruning metrics ...")
        self.prune_metrics()
        if attack and self.attack_method is not None and self.attack_kwargs is not None:
            print("Getting adversarial accuracy ...")
            self.adv_acc()
        return self.print_results()


if __name__ == "__main__":
    path = "experiments/experiment_12/googlenet/CIFAR10/googlenet_GreedyPGDGlobalMagGrad_2_compression_5_finetune_iterations/prune/checkpoints/checkpoint-5.pt"
    model_type = "googlenet"
    gpu = 0
    evaluator = Model_Evaluator(
        model_type=model_type, model_path=pathlib.Path(path), gpu=gpu
    )
    evaluator.evaluate()
