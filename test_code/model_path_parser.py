from pathlib import Path
import os

experiment_number = None
dataset = None
model_type = "googlenet"
quantization = 4
prune_method = "RandomPruning"
attack_method = None
finetune_epochs = 40
prune_compression = 2

p = Path(r"\aicas\experiments\experiment_1\googlenet\CIFAR10\googlenet_RandomPruning_2_compression_40_finetune_iterations\prune\checkpoints\checkpoint-40.pt")

model_path = p

m_path = str(p)
sep = '\\' if '\\' in m_path else '/'

if experiment_number:
    if f"experiment_{m_pathexperiment_number}" not in m_path:
        raise ValueError(f"Provided experiment number {experiment_number} but provided model path"
                         f"\n{model_path} does not include this experiment number.")
else:
    experiment_number = int(m_path[m_path.find("experiment_"):].split("_")[1].split(sep)[0])

if model_type:
    if model_type not in m_path:
        raise ValueError(f"Provided model type {model_type} but provided model path"
                         f"\n{model_path} does not include this model_type.")
else:
    model_type = m_path[m_path.find(f"experiment_{experiment_number}"):].split(sep)[1]

if dataset:
    if dataset not in m_path:
        raise ValueError(f"Provided dataset {dataset} but provided model path"
                         f"\n{model_path} does not include this dataset.")
else:
    dataset = m_path[m_path.find(model_type):].split(sep)[1]

if quantization:
    # only throw an error if there is a different quantization already applied.
    if "quantization" in m_path:
        if f"{quantization}_quantization" not in m_path:
            raise ValueError(f"Provided quantization {quantization} but provided model path"
                             f"\n{model_path} does not include this quantization.")
else:
    loc = m_path.find("quantization")
    if loc >= 0:
        quantization = int(m_path[:loc].split("_")[-2])
    else:
        quantization = None

# indicates whether or not the model from model_path is already pruned.
already_pruned = False

if prune_compression:
    # only throw an error if there is a different compression already applied.
    if "compression" in m_path:
        if f"{prune_compression}_compression" not in m_path:
            raise ValueError(f"Provided pruning compression {prune_compression} but provided model path"
                             f"\n{model_path} does not include this prune compression.")
        else:
            already_pruned = True
else:
    loc = m_path.find("compression")
    if loc >= 0:
        prune_compression = int(m_path[:loc].split("_")[-2])
    else:
        prune_compression = None

if prune_method:
    # only throw an error if there is a different pruning already applied
    #   (checked using already_pruned variable)
    if prune_method not in m_path and already_pruned:
        raise ValueError(f"Provided pruning method {prune_method} but provided model path"
                         f"\n{model_path} does not include this pruning method.")
else:
    if prune_compression:
        prune_method = m_path[:m_path.find(f"{prune_compression}_compression")].split("_")[-2]
    else:
        prune_method = None

if finetune_epochs:
    # only throw an error if there is a different finetuning already applied
    #   (checked using already_pruned variable)
    if f"{finetune_epochs}_finetune_iterations" not in m_path and already_pruned:
        raise ValueError(f"Provided finetune epochs {finetune_epochs} but provided model path"
                         f"\n{model_path} does not include this finetune_epochs.")
else:
    if prune_method:
        finetune_epochs = int(m_path[:m_path.find("finetune")].split("_")[-2])
    else:
        finetune_epochs = None

if quantization:
    # only throw an error if there is a different quantization already applied
    if "quantization" in m_path:
        if f"{quantization}_quantization" not in m_path:
            raise ValueError(f"Provided quantization {quantization} but provided model path"
                             f"\n{model_path} does not include this quantization.")
else:
    loc = m_path.find("quantization")
    if loc >= 0:
        quantization = int(m_path[:loc].split("_")[-2])
    else:
        quantization = None

print(m_path)
print(f"Experiment_number: {experiment_number}, type: {type(experiment_number)}")
print(f"dataset: {dataset}, type: {type(dataset)}")
print(f"model type: {model_type}, type: {type(model_type)}")
print(f"quantization: {quantization}, type: {type(quantization)}")
print(f"prune method: {prune_method}, type: {type(prune_method)}")
print(f"attack_method {attack_method}, type: {type(attack_method)}")
print(f"finetune_epochs: {finetune_epochs}, type: {type(finetune_epochs)}")
print(f"prune compression: {prune_compression}, type: {type(prune_compression)}")
