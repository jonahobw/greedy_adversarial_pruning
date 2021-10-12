from shrinkbench.experiment import TrainingExperiment

import os
from pathlib import Path
from multiprocessing import freeze_support

path = Path(os.getcwd()).parent.absolute() / "aicas/datasets"
os.environ['DATAPATH'] = str(path)
freeze_support()

exp = TrainingExperiment(dataset='CIFAR10', model="resnet20", dl_kwargs={"num_workers":1}, train_kwargs={"epochs": 1})
exp.run()