from shrinkbench.experiment import TrainingExperiment, PruningExperiment

import os
from pathlib import Path
from multiprocessing import freeze_support

path = Path(os.getcwd()).parent.absolute() / "aicas/datasets"
os.environ["DATAPATH"] = str(path)
exp_path = (
    Path.cwd() / "experiments" / "experiment_0" / "ResNet18" / "cifar10" / "resnet18"
)


def expr():
    exp2 = PruningExperiment(
        dataset="CIFAR10",
        model="resnet20",
        strategy="RandomPruning",
        compression=2,
        train_kwargs={"epochs": 1},
        path=exp_path,
    )
    exp2.run()


if __name__ == "__main__":
    print(Path.cwd())
    # freeze_support()
    # expr()
