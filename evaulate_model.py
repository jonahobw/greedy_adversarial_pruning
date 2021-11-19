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

from shrinkbench.experiment import DNNExperiment
from shrinkbench import models
from shrinkbench.models.head import mark_classifier
from shrinkbench.metrics import model_size, flops, accuracy, correct
from shrinkbench.util import OnlineStats

attacks = {'pgd': projected_gradient_descent}

class Model_Evaluator(DNNExperiment):
    dl_kwargs = {
        "batch_size": 128,
        "pin_memory": False,
        "num_workers": 4,
    }

    def __init__(self, model_type, model_path, dataset='CIFAR10', gpu=None, seed=42, dl_kwargs: {} = None):
        super().__init__(seed=seed)

        self.fix_seed(seed, deterministic=True)

        # set environment variable to be used by shrinkbench
        os.environ["DATAPATH"] = str(pathlib.Path.cwd() / 'datasets')

        b = pathlib.Path.cwd() / pathlib.Path(model_path)
        self.path = b
        if os.name != "nt":
            self.path = pathlib.Path(b.as_posix())
        print(f"Provided model path: {self.path}")
        if not self.path.exists():
            raise FileNotFoundError(f"Model path {path} not found.")
        self.resume = self.path

        self.gpu = gpu
        self.dataset=dataset
        self.model_type = model_type
        self.model_path = model_path
        if dl_kwargs:
            self.dl_kwargs = self.dl_kwargs.update(dl_kwargs)
        self.build_dataloader(dataset=dataset, **self.dl_kwargs)
        self.build_model(self.model_type, pretrained=False, resume=self.model_path, dataset=dataset)

    def clean_acc(self):
        res = list(accuracy(model=self.model, dataloader=self.train_acc_dl, topk=(1, 5)))
        self.clean_train_acc1 = res[0]
        self.clean_train_acc5 = res[1]
        res = list(accuracy(model=self.model, dataloader=self.val_dl, topk=(1, 5)))
        self.clean_val_acc1 = res[0]
        self.clean_val_acc5 = res[1]

    def prune_metrics(self):
        """Collect the pruning metrics."""
        # Model Size
        size, size_nz = model_size(self.model)
        self.model_size = size
        self.model_size_nz = size_nz
        self.compression_ratio = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        self.flops = ops
        self.flops_nz = ops_nz
        self.theoretical_speedup = ops / ops_nz

    def adv_acc(self, train=True, attack_name='pgd', attack_kwargs={"eps": 2 / 255, "eps_iter": 0.001, "nb_iter": 10, "norm": np.inf}):

        self.model.eval()

        if train:
            dl = self.train_dl
            data = "train"
        else:
            dl = self.val_dl
            data = "test"

        results = {"dataset": data, "inputs_tested": 0}

        clean_acc1 = OnlineStats()
        clean_acc5 = OnlineStats()
        adv_acc1 = OnlineStats()
        adv_acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{attack_name} on {data} dataset")

        for i, (x, y) in enumerate(epoch_iter, start=1):
            x, y = x.to(self.device), y.to(self.device)
            x_adv = attacks[attack_name](self.model, x, **attack_kwargs)
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
        for name in ['clean_train_acc1', 'clean_train_acc5', 'clean_val_acc1', 'clean_val_acc5', 'model_size',
                     'model_size_nz', 'compression_ratio', 'flops', 'flops_nz', 'theoretical_speedup',
                     'adv_results']:
            if hasattr(self, name):
                metrics[name] = getattr(self, name)

        print(json.dumps(metrics, indent=4))

    def run(self, attack=False):
        print("Evaluating model ...\nGetting clean accuracy ...")
        self.clean_acc()
        print("Getting pruning metrics ...")
        self.prune_metrics()
        if attack:
            print("Getting adversarial accuracy ...")
            self.adv_acc()
        self.print_results()


if __name__ == '__main__':
    path = 'experiments/experiment_12/googlenet/CIFAR10/googlenet_GreedyPGDGlobalMagGrad_2_compression_5_finetune_iterations/prune/checkpoints/checkpoint-5.pt'
    model_type = 'googlenet'
    gpu = 0
    evaluator = Model_Evaluator(model_type=model_type, model_path=pathlib.Path(path), gpu=gpu)
    evaluator.run()
