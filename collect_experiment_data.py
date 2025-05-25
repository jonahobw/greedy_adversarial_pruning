"""
Aggregates, formats, and exports experiment results.

Includes utilities for collecting accuracy, pruning, and adversarial attack metrics
across multiple experiments.
"""
from pathlib import Path
import json
import collections
import pandas as pd
import numpy as np
import datetime
import csv

from experiment_utils import find_recent_file


def flatten_dict(d, parent_key="", sep="_"):
    """
    Recursively flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string for recursion.
        sep (str): Separator between parent and child keys.

    Returns:
        dict: Flattened dictionary with compound keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_json(path, prefix, vals=None, flat=True):
    """
    Reads the latest JSON file in a directory with a given prefix and returns selected values.

    Args:
        path (str or Path): Directory to search for the file.
        prefix (str): Prefix of the file to find.
        vals (list, optional): Keys to extract from the JSON.
        flat (bool): Whether to flatten nested dictionaries.

    Returns:
        dict: Dictionary of requested values from the JSON file.
    """
    params_file = find_recent_file(path, prefix)
    if params_file == -1:
        # no file found
        return {x: None for x in vals}
    with open(str(params_file), "r") as file:
        params = json.load(file)
    if vals:
        params = {x: params[x] for x in vals}
    if flat:
        params = flatten_dict(params)
    return params


def read_experiment_params(path, vals=None, flat=True):
    """
    Reads the latest experiment parameter file in a directory.

    Args:
        path (str or Path): Directory containing the experiment parameter file.
        vals (list, optional): Keys to extract from the file.
        flat (bool): Whether to flatten nested dictionaries.

    Returns:
        dict: Dictionary of experiment parameters.
    """
    if vals is None:
        vals = [
            "debug",
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
            "save_one_checkpoint",
            "seed",
            "train_from_scratch",
            "train_kwargs",
            "prune_kwargs",
        ]

    return read_json(path=path, prefix="experiment_params", vals=vals, flat=flat)


def read_pruning_metrics(path, vals=None, flat=True):
    """
    Reads pruning metrics from the most recent metrics JSON in the prune folder.

    Args:
        path (str or Path): Path to the experiment folder.
        vals (list, optional): Keys to extract from the metrics file.
        flat (bool): Whether to flatten nested dictionaries.

    Returns:
        dict: Dictionary of pruning metrics.
    """
    if vals is None:
        vals = [
            "size",
            "size_nz",
            "compression_ratio",
            "flops",
            "flops_nz",
            "theoretical_speedup",
            "loss",
        ]

    prune_folder = str(Path(path) / "prune")
    return read_json(path=prune_folder, prefix="metrics", vals=vals, flat=flat)


def read_attack_metrics(path, vals=None, flat=True, attack="pgd"):
    """
    Reads adversarial attack metrics from the most recent attack results JSON.

    Args:
        path (str or Path): Path to the experiment folder.
        vals (list, optional): Keys to extract from the results file.
        flat (bool): Whether to flatten nested dictionaries.
        attack (str): Attack method name (e.g., 'pgd').

    Returns:
        dict: Dictionary of attack metrics.
    """
    if vals is None:
        vals = ["results"]
    attack_folder = str(Path(path) / "attacks" / attack)
    return  read_json(
        path=attack_folder, prefix="train_attack_results", vals=vals, flat=flat
    )


def read_transfer_attack_metrics(path, vals=None, flat=True, attack="pgd"):
    """
    Reads transfer attack metrics from the most recent transfer attack results JSON.

    Args:
        path (str or Path): Path to the experiment folder.
        vals (list, optional): Keys to extract from the results file.
        flat (bool): Whether to flatten nested dictionaries.
        attack (str): Attack method name (e.g., 'pgd').

    Returns:
        dict: Dictionary of transfer attack metrics.
    """
    if vals is None:
        vals = ["transfer_results", "transfer_model_path"]
    attack_folder = str(Path(path) / "attacks" / attack / "transfer_attack")
    return read_json(
        path=attack_folder, prefix="train_attack_results", vals=vals, flat=flat
    )


def read_accuracy_results(path, vals=None):
    """
    Reads accuracy results from the most recent logs in the prune or train folder.

    Args:
        path (str or Path): Path to the experiment folder.
        vals (list, optional): Keys to extract from the logs file.

    Returns:
        dict: Dictionary of clean accuracy results.
    """
    if vals is None:
        vals = ["train_acc1", "train_acc5", "val_acc1", "val_acc5"]

    acc_path = Path(path) / "prune"
    if not acc_path.exists():
        acc_path = Path(path) / "train"
        if not acc_path.exists():
            raise FileNotFoundError(f"Could not find train or prune folder in {path}")

    csv_file = find_recent_file(acc_path, "logs")
    df = pd.read_csv(csv_file, usecols=vals)
    # this assumes no best_model_metric
    params = df.iloc[-1]
    params = params.to_dict()

    return {f"clean_{x}": params[x] for x in params}


def collect_one_experiment_data(path):
    """
    Collects all relevant metrics and parameters for a single experiment folder.

    Args:
        path (str or Path): Path to the experiment folder.

    Returns:
        dict: Aggregated results from experiment parameters, pruning, attack, transfer, and accuracy.
    """
    experiment_params = read_experiment_params(path)
    pruning_metrics = read_pruning_metrics(path)
    attack_metrics = read_attack_metrics(
        path, attack=experiment_params["attack_method"]
    )
    transfer_attack_metrics = read_transfer_attack_metrics(
        path, attack=experiment_params["attack_method"]
    )
    accuracy_results = read_accuracy_results(path)

    results = {
        **experiment_params,
        **pruning_metrics,
        **attack_metrics,
        **transfer_attack_metrics,
        **accuracy_results,
    }
    return results


def find_experiment_folders(exp_folder, models, dataset="CIFAR10"):
    """
    Finds all experiment subfolders for a given set of models and dataset.

    Args:
        exp_folder (Path): Path to the experiment root folder.
        models (list): List of model names.
        dataset (str): Dataset name (default: 'CIFAR10').

    Returns:
        list: List of Path objects for each experiment folder.
    """
    experiment_folders = []
    for model in models:
        model_path = exp_folder / model / dataset
        experiment_folders.extend(list(model_path.glob(f"{model}*")))
    return experiment_folders


def generate_csv(experiment_number, models=None, timestamp=True):
    """
    Aggregates results from all experiment folders and writes them to a CSV file.

    Args:
        experiment_number (int): The experiment identifier.
        models (list, optional): List of model names. Defaults to common models.
        timestamp (bool): Whether to append a timestamp to the CSV filename.
    """
    if models is None:
        models = ["resnet20", "googlenet", "vgg_bn_drop", "mobilenet_v2"]

    experiment_x = Path.cwd() / "experiments" / f"experiment_{experiment_number}"
    experiment_folders = find_experiment_folders(experiment_x, models)

    csv_name = "collected_results.csv"
    if timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_name = f"collected_results_{timestamp}.csv"
    csv_path = experiment_x / csv_name

    # want ordered & unique data type
    headers = {}
    data = []
    for exp in experiment_folders:
        results = collect_one_experiment_data(exp)
        headers.update({x: None for x in results})
        data.append(results)

    with open(csv_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=headers.keys(), lineterminator="\n")
        writer.writeheader()
        for x in data:
            writer.writerow(x)


def generate_csv_new_workflow(experiment_number, models=None, timestamp=True):
    """
    Aggregates results using the new workflow and writes them to a CSV file.

    Args:
        experiment_number (int): The experiment identifier.
        models (list, optional): List of model names. Defaults to common models.
        timestamp (bool): Whether to append a timestamp to the CSV filename.

    Returns:
        Path: Path to the generated CSV file.
    """
    if models is None:
        models = ["resnet20", "googlenet", "vgg_bn_drop", "mobilenet_v2"]

    experiment_x = Path.cwd() / "experiments" / f"experiment_{experiment_number}"
    experiment_folders = find_experiment_folders(experiment_x, models)

    csv_name = "collected_results.csv"
    if timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_name = f"collected_results_{timestamp}.csv"
    csv_path = experiment_x / csv_name

    # want ordered & unique data type
    headers = {}
    data = []
    for exp in experiment_folders:
        results = read_json(exp, prefix="experiment_results_", flat=True)
        headers.update({x: None for x in results})
        data.append(results)

    # make sure all data cases have the same headers
    for x in data:
        for key in headers:
            if key not in x:
                x[key] = None

    with open(csv_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=headers.keys(), lineterminator="\n")
        writer.writeheader()
        for x in data:
            writer.writerow(x)

    return csv_path


def fill_empty_csv(experiment, path=None):
    """
    Fills in empty prune attack parameters and transfer attacks for models which were not pruned.
    The transfer attack results will simply be the identity function.

    Args:
        experiment (int): The experiment identifier.
        path (Path, optional): Path to the CSV file. If None, finds the most recent.

    Returns:
        Path: Path to the new formatted CSV file.
    """
    experiment_folder = Path.cwd() / "experiments" / f"experiment_{experiment}"

    if path is None:
        path = find_recent_file(experiment_folder, prefix="collected_results")

    df = pd.read_csv(path)

    convert_results = [
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
        "model_file_size",
        "attack_results_inputs_tested",
        "attack_results_clean_acc1",
        "attack_results_clean_acc5",
        "attack_results_adv_acc1",
        "attack_results_adv_acc5",
        "attack_results_runtime",
    ]

    # Fill missing prune_* columns with train_* values if not pruned
    df_train_scratch = df[df["prune_model_file_size"].isna()]
    for x in convert_results:
        df[f"prune_{x}"] = np.where(
            df[f"prune_{x}"].isna(), df[f"train_{x}"], df[f"prune_{x}"]
        )

    more_convert_results = [
        ("attack_transfer_both_correct1", "attack_results_adv_acc1"),
        ("attack_transfer_both_correct5", "attack_results_adv_acc5"),
        ("attack_transfer_transfer_correct1", "attack_results_adv_acc1"),
        ("attack_transfer_transfer_correct5", "attack_results_adv_acc5"),
        ("attack_transfer_target_correct1", "attack_results_adv_acc1"),
        ("attack_transfer_target_correct5", "attack_results_adv_acc5"),
    ]

    # Fill missing transfer attack columns for pruned/quantized models
    for prefix_empty, prefix_replace in [
        ("prune_", "train_"),
        ("quantize_", "quantize_"),
    ]:
        for empty, replace in more_convert_results:
            df[f"{prefix_empty}{empty}"] = np.where(
                df[f"{prefix_empty}{empty}"].isna(),
                df[f"{prefix_replace}{replace}"],
                df[f"{prefix_empty}{empty}"],
            )

    new_csv = path.parent / f"formatted_{path.name}"
    df.to_csv(new_csv, index=False)
    return new_csv


def retrieve_formatted_csv_df(experiment, cols=None, regenerate=True):
    """
    Retrieves a formatted DataFrame for a given experiment, regenerating files if needed.

    Args:
        experiment (int): The experiment identifier.
        cols (list, optional): Columns to include in the DataFrame.
        regenerate (bool): Whether to regenerate the CSV files.

    Returns:
        pd.DataFrame: DataFrame with the requested columns.
    """
    experiment_folder = Path.cwd() / "experiments" / f"experiment_{experiment}"
    if not regenerate:
        raw_results = find_recent_file(experiment_folder, prefix="collected_results")
        if isinstance(raw_results, int):
            generate_csv_new_workflow(experiment)
        file = find_recent_file(experiment_folder, prefix="formatted_collected_results")
        if isinstance(file, int):
            file = fill_empty_csv(experiment)
    else:
        csvs = list(experiment_folder.glob("*.csv"))
        for f in csvs:
            f.unlink()
        generate_csv_new_workflow(experiment)
        file = fill_empty_csv(experiment)

    return pd.read_csv(file, usecols=cols)


def format_data_for_plot(experiment, metrics, pruning=None, models=None):
    """
    Formats experiment data for plotting, returning grouped and ungrouped DataFrames.

    Args:
        experiment (int): The experiment identifier.
        metrics (list): List of metric column names to include.
        pruning (list, optional): List of pruning strategies.
        models (list, optional): List of model names.

    Returns:
        tuple: (grouped_data, ungrouped_data) where each is a dict by model name.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    if models is None:
        models = ["resnet20", "googlenet", "vgg_bn_drop", "mobilenet_v2"]

    if pruning is None:
        pruning = [
            "RandomPruning",
            "GlobalMagWeight",
            "LayerMagWeight",
            "GlobalMagGrad",
            "LayerMagGrad",
            "GreedyPGDGlobalMagGrad",
            "GreedyPGDLayerMagGrad",
        ]

    cols = [
        "name",
        "model_type",
        "prune_method",
        "prune_compression",
    ]
    cols.extend(metrics)
    df_orig = retrieve_formatted_csv_df(experiment, cols=cols)

    final_data = {}
    ungrouped_data = {}

    for model in models:
        df = df_orig[df_orig["model_type"] == model]
        zero_compression_metrics = {}
        for metric in metrics:
            zero_compression_metrics[metric] = df[df["name"] == model][metric].item()
        temp = {**zero_compression_metrics, "prune_compression": 1}
        for prune_strat in pruning:
            df = df.append({**temp, "prune_method": prune_strat}, ignore_index=True)
        df = df.sort_values("prune_compression")
        ungrouped_data[model] = df
        final_data[model] = df.groupby(["prune_method"])
    return final_data, ungrouped_data


def format_multiple_experiment_data_for_plot(
    experiments, metrics, pruning=None, models=None
):
    """
    Combines data from multiple experiments for plotting, averaging metrics and computing std dev.

    Args:
        experiments (list): List of experiment identifiers.
        metrics (list): List of metric column names to include.
        pruning (list, optional): List of pruning strategies.
        models (list, optional): List of model names.

    Returns:
        dict: Dictionary of DataFrames by model, with mean and std for each metric.
    """
    if not isinstance(experiments, list):
        experiments = [experiments]

    if not isinstance(metrics, list):
        metrics = [metrics]

    if models is None:
        models = ["resnet20", "googlenet", "vgg_bn_drop", "mobilenet_v2"]

    experiment_data = []
    for experiment_x in experiments:
        experiment_data.append(
            format_data_for_plot(
                experiment=experiment_x, metrics=metrics, pruning=pruning, models=models
            )[1]
        )
        a = 0  # placeholder, not used

    # Combine model dataframes and compute mean/std
    model_dfs = {}
    for model in models:
        temp = pd.concat([x[model] for x in experiment_data])
        model_dfs[model] = (
            temp.groupby(
                ["prune_method", "prune_compression"],
                as_index=False,
            )
            .agg({metrc: ["mean", "std"] for metrc in metrics})
            .sort_values("prune_compression")
        )
    return model_dfs


if __name__ == "__main__":
    # Example: combine and format data from multiple experiments for plotting
    format_multiple_experiment_data_for_plot(
        experiments=[10, 11, 12],
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
    )
    # Other metrics:
    # 'quantize_attack_transfer_target_correct_1',
    # 'quantize_attack_results_adv_acc1'
