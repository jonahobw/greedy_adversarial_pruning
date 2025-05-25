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
    Reads the latest file in path that starts with the prefix, returning a dictionary of the values.

    :param path: folder path, the target file is the most recent file in this folder
                named "<prefix>*.json"
    :param vals: list of the values to read from the file.
    :return: a dictionary of values from the latest experiment parameter file
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
    Reads the latest experiment parameter file in path, and returns a dictionary of the values.

    :param path: folder path, the experiment parameter file is in this folder and is named
            "experiment_params_<timestamp>.json"
    :param vals: list of the values to read from the file.
    :param flat: if true, then will flatten out nested dicts
    :return: a dictionary of values from the latest experiment parameter file
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

    params = read_json(path=path, prefix="experiment_params", vals=vals, flat=flat)

    return params


def read_pruning_metrics(path, vals=None, flat=True):
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
    params = read_json(path=prune_folder, prefix="metrics", vals=vals, flat=flat)

    # # rename args
    # if "size" in params:
    #     params["model_size"] = params["size"]
    #     params.pop("size")
    # else:
    #     params["model_size"] = None
    # if "size_nz" in params:
    #     params["model_size_nz"] = params["size_nz"]
    #     params.pop("size_nz")
    # else:
    #     params["model_size_nz"] = None

    return params


def read_attack_metrics(path, vals=None, flat=True, attack="pgd"):
    if vals is None:
        vals = ["results"]
    attack_folder = str(Path(path) / "attacks" / attack)
    params = read_json(
        path=attack_folder, prefix="train_attack_results", vals=vals, flat=flat
    )

    # # format args
    # if "results" in params:
    #     if "adv_acc1" in params["results"]:
    #         params["adv_acc1"] = params["results"]["adv_acc1"]
    #     if "adv_acc5" in params["results"]:
    #         params["adv_acc5"] = params["results"]["adv_acc5"]
    #     params.pop("results")
    # else:
    #     params["adv_acc1"] = None
    #     params["adv_acc5"] = None
    #
    # if flat:
    #     params = flatten_dict(params)
    return params


def read_transfer_attack_metrics(path, vals=None, flat=True, attack="pgd"):
    if vals is None:
        vals = ["transfer_results", "transfer_model_path"]
    attack_folder = str(Path(path) / "attacks" / attack / "transfer_attack")
    params = read_json(
        path=attack_folder, prefix="train_attack_results", vals=vals, flat=flat
    )

    # # format args
    # if params"transfer_results" is not None:
    #     if "transfer_runtime" in params["transfer_results"]:
    #         params["transfer_results"].pop("transfer_runtime")
    # elif params["transfer_results"] is None:
    #     params["transfer_results"] = {
    #                         "inputs_tested": None,
    #                         "both_correct1": None,
    #                         "both_correct5": None,
    #                         "transfer_correct1": None,
    #                         "transfer_correct5": None,
    #                         "target_correct1": None,
    #                         "target_correct5": None,
    #                     }
    #
    # if flat:
    #     params = flatten_dict(params)
    return params


def read_accuracy_results(path, vals=None):
    """If model has been pruned, read from pruning logs, else from training logs."""

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

    experiment_folders = []

    for model in models:
        model_path = exp_folder / model / dataset
        experiment_folders.extend(list(model_path.glob(f"{model}*")))

    return experiment_folders


def generate_csv(experiment_number, models=None, timestamp=True):
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
    Fills in empty prune attack parameters and prune/quantized transfer attacks for
    models which were not pruned.  The transfer attack results will simply be the identity
    function.

    :param experiment:
    :param path:
    :return:
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

    # columns attack_transfer_transfer_model, attack_transfer_transfer_runtime, and
    #   attack_transfer_inputs_tested are left blank

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
    # with open(new_csv, 'w') as f:
    df.to_csv(new_csv, index=False)
    return new_csv


def retrieve_formatted_csv_df(experiment, cols=None, regenerate=True):
    """
    If regenerate is true, regenerates the files and removes old ones.
    :param experiment:
    :param cols:
    :param replace:
    :return:
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
    Returns a tuple ({model: groupby dataframe}, {model: big dataframe})

    The first item in the dict is for easy plotting, the second is for combining
    with other experiments

    Can plot the groupby dataframe using

    for pruning_strat, data in groupby_dataframe:
        x_axis = data['prune_compression']
        y_axis = data[metric]
        label = pruning_strat
        plt.plot(x_axis, y_axis, label=label)

    :param experiment: which experiment to get the data from
    :param metrics: list of columns which you want to plot on y axis.
    :param pruning: list of pruning strategies on which collect data
    :param models: list of models on which to collect data
    :return:
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
        # for key, group in a:
        #     print(key)
        #     print("\n\n\n")
        #     print(group)
    return final_data, ungrouped_data


def format_multiple_experiment_data_for_plot(
    experiments, metrics, pruning=None, models=None
):
    """
    Combines the data from multiple experiment runs into one result of the same format
    as format_data_for_plot(), except that the metrics columns are the averages of the
    results from each experiment, and there is an additional column for each metric which
    is the standard deviation for that metric.

    :param experiments:
    :param metrics:
    :param pruning:
    :param models:
    :return:
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
        a = 0

    # combine model dataframes
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
    # generate_csv_new_workflow(10)
    # clean_prune_acc(10)
    format_multiple_experiment_data_for_plot(
        experiments=[10, 11, 12],
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
    )
    #'prune_attack_transfer_target_correct_1', 'quantize_attack_transfer_target_correct_1',
    # 'quantize_attack_results_adv_acc1'
