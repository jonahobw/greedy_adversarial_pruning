"""
Data analysis and plotting utilities for experiments.

Provides functions to visualize and compare the effects of pruning, quantization,
and adversarial attacks on model performance, as described in the methodology.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from experiment_utils import find_recent_file
from collect_experiment_data import format_multiple_experiment_data_for_plot


def experiment_csv(experiment_number, cols=None):
    experiment_path = Path.cwd() / "experiments" / f"experiment_{experiment_number}"
    csv_file = find_recent_file(experiment_path, prefix="collected_results")
    return pd.read_csv(csv_file, usecols=cols)


def plot(experiment_number, models=None, pruning=None):
    """
    Plot a grid.

    first row is the clean train acc by compression ratio, and each pruning strategy has a line,
    and each model has a graph
    second row is the adv train acc by compression ratio, and each pruning strategy has a line,
    and each model has a graph
    3rd row is the transfer train acc by compression ratio, and each pruning strategy has a line,
    and each model has a graph
    :return:
    """
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
        "clean_val_acc1",
        "results_adv_acc1",
        "transfer_results_target_correct1",
    ]

    y_axis = ["clean_val_acc1", "results_adv_acc1", "transfer_results_target_correct1"]

    df_orig = experiment_csv(experiment_number, cols=cols)

    fig, ax = plt.subplots(len(y_axis), len(models), sharex="none", sharey="row")
    fig.set_size_inches(18.5, 10.5)
    for j in range(len(y_axis)):
        metric = y_axis[j]
        for i in range(len(models)):
            model = models[i]
            df = df_orig[df_orig["model_type"] == model]
            df_0_compression_acc = df[df["name"] == model][metric].item()
            df = df[[metric, "prune_compression", "prune_method"]]
            temp = {metric: df_0_compression_acc, "prune_compression": 1}
            for prune_strat in pruning:
                df = df.append({**temp, "prune_method": prune_strat}, ignore_index=True)
            df = df.sort_values("prune_compression")
            print(df.groupby(["prune_method"]))
            for key, grp in df.groupby(["prune_method"]):
                if key in pruning:
                    grp.plot(
                        ax=ax[j, i],
                        kind="line",
                        x="prune_compression",
                        y=metric,
                        label=key,
                    )
            ax[j, i].set_xticks(grp["prune_compression"])
            if j == 0:
                ax[j, i].set_title(f"{model} Metrics by Compression Ratio\n{metric}")
            else:
                ax[j, i].set_title(f"{metric}")
            if j != len(y_axis) - 1:
                ax[j, i].xaxis.label.set_visible(False)
            ax[j, i].legend(prop={"size": 8})

    # don't know why but these need to be here to get the labels
    for j in range(len(y_axis)):
        for i in range(len(models)):
            ax[j, i].yaxis.set_tick_params(labelleft=True)
    plt.show()


def simple_mean_std():
    x = [1, 2, 3, 4]
    y_mean = [1, 1.5, 1.75, 1.85]
    y_std = [0.1, 0.2, 0.3, 0.4]
    plt.errorbar(x, y_mean, yerr=[2 * x for x in y_std], fmt="o-")
    plt.show()


def plot_all_experiment_data(
    metrics, experiments=None, models=None, pruning=None, error_bars=False, save=None
):
    """
    Makes a plot of dimensions (# of metrics) x (# of models).  Each subplot has
    averaged data from the experiments with error bars showing +-2std, and each
    subplot has 1 line per pruning method.  The x axis is prune compression and
    the y axis is the metric.

    :param metrics:
    :param experiments:
    :param models:
    :param pruning:
    :return:
    """

    metrics_mapping = {
        "prune_clean_val_acc1": {"y_ax": [0, 1], "label": "Clean Test Accuracy"},
        "quantize_clean_val_acc1": {"y_ax": [0, 1], "label": "Quantize Clean Accuracy"},
        "prune_attack_results_adv_acc1": {
            "y_ax": [0, 0.4],
            "label": "Adversarial Accuracy",
        },
        "prune_attack_transfer_target_correct1": {
            "y_ax": [0, 0.8],
            "label": "Transfer Attack Accuracy",
        },
        "quantize_attack_transfer_target_correct1": {
            "y_ax": [0, 0.8],
            "label": "Quantize Transfer Attack Accuracy",
        },
        "quantize_attack_results_adv_acc1": {
            "y_ax": [0, 0.4],
            "label": "Quantize Adversarial Accuracy",
        },
    }

    if experiments is None:
        experiments = [10, 11, 12]

    model_mapping = {
        "resnet20": "ResNet20",
        "googlenet": "GoogLeNet",
        "vgg_bn_drop": "VGG",
        "mobilenet_v2": "MobileNetV2",
    }

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

    prune_mapping = {
        "RandomPruning": "RandomPruning",
        "GlobalMagWeight": "GlobalParamMag",
        "LayerMagWeight": "LayerParamMag",
        "GlobalMagGrad": "GlobalParamGrad",
        "LayerMagGrad": "LayerParamGrad",
        "GreedyPGDGlobalMagGrad": "GlobalGAP",
        "GreedyPGDLayerMagGrad": "LayerGAP",
    }

    data = format_multiple_experiment_data_for_plot(
        experiments=experiments, metrics=metrics, pruning=pruning, models=models
    )

    fig, ax = plt.subplots(len(metrics), len(models), sharex="none", sharey="none")
    fig.set_size_inches(18.5, 10.5)
    for j in range(len(metrics)):
        metric = metrics[j]
        for i in range(len(models)):
            model = models[i]
            df = data[model]
            ax[j, i].set_ylim(metrics_mapping[metric]["y_ax"])
            for prune_strat in pruning:
                df_prune = df[df["prune_method"] == prune_strat]
                x = df_prune["prune_compression"]
                y = df_prune[metric]["mean"]
                # y_err = 2* df_prune[metric]['std'] if error_bars else None #95% confidence
                y_err = (
                    df_prune[metric]["std"] if error_bars else None
                )  # 68% confidence
                if prune_strat == "GreedyPGDGlobalMagGrad":
                    ax[j, i].errorbar(
                        x, y, yerr=y_err, label=prune_mapping[prune_strat], linewidth=3
                    )
                else:
                    ax[j, i].errorbar(
                        x, y, yerr=y_err, label=prune_mapping[prune_strat]
                    )
                # ax[j, i].errorbar(x, y, yerr=y_err, label=prune_strat)
                # ax[j, i].errorbar(x, y, yerr=y_err, fmt='o-', label=prune_strat)
            ax[j, i].set_xscale("log", base=2)
            ax[j, i].set_xticks(x)
            ax[j, i].xaxis.set_major_formatter(ScalarFormatter())
            # if j == len(metrics) - 1:
            # ax[j, i].xaxis.set_major_formatter(ScalarFormatter())
            # else:
            #     ax[j, i].xaxis.set_ticklabels([])
            if i == 0 and j == 0:
                ax[j, i].legend(prop={"size": 10}, framealpha=0.0)
            if j == 0:
                ax[j, i].set_title(model_mapping[model])
                # ax[j, i].set_title(f"{model_mapping[model]}\n"
                #                    f"{metrics_mapping[metric]['label']}")
            if j == len(metrics) - 1:
                ax[j, i].set_xlabel("Prune Compression")
            if i == 0:
                ax[j, i].set_ylabel(metrics_mapping[metric]["label"])

    if save:
        parent = Path.cwd() / "plots"
        parent.mkdir(exist_ok=True)
        save_path = parent / save
        plt.savefig(save_path)
    plt.show()


def plot_all_experiment_data_transpose(
    metrics, experiments=None, models=None, pruning=None, error_bars=False
):
    """
    Makes a plot of dimensions (# of models) x (# of metrics).  Each subplot has
    averaged data from the experiments with error bars showing +-2std, and each
    subplot has 1 line per pruning method.  The x axis is prune compression and
    the y axis is the metric.

    :param metrics:
    :param experiments:
    :param models:
    :param pruning:
    :return:
    """

    metrics_mapping = {
        "prune_clean_val_acc1": {"y_ax": [0, 1], "label": "Clean Test Accuracy"},
        "quantize_clean_val_acc1": {"y_ax": [0, 1], "label": "Quantize Clean Accuracy"},
        "prune_attack_results_adv_acc1": {
            "y_ax": [0, 0.4],
            "label": "Adversarial Accuracy",
        },
        "prune_attack_transfer_target_correct1": {
            "y_ax": [0, 0.8],
            "label": "Transfer Attack Accuracy",
        },
        "quantize_attack_transfer_target_correct1": {
            "y_ax": [0, 0.8],
            "label": "Quantize Transfer Attack Accuracy",
        },
        "quantize_attack_results_adv_acc1": {
            "y_ax": [0, 0.4],
            "label": "Quantize Adversarial Accuracy",
        },
    }

    if experiments is None:
        experiments = [10, 11, 12]

    model_mapping = {
        "resnet20": "ResNet20",
        "googlenet": "GoogLeNet",
        "vgg_bn_drop": "VGG",
        "mobilenet_v2": "MobileNetV2",
    }

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

    prune_mapping = {
        "RandomPruning": "RandomPruning",
        "GlobalMagWeight": "GlobalParamMag",
        "LayerMagWeight": "LayerParamMag",
        "GlobalMagGrad": "GlobalParamGrad",
        "LayerMagGrad": "LayerParamGrad",
        "GreedyPGDGlobalMagGrad": "GlobalGAP",
        "GreedyPGDLayerMagGrad": "LayerGAP",
    }

    data = format_multiple_experiment_data_for_plot(
        experiments=experiments, metrics=metrics, pruning=pruning, models=models
    )

    fig, ax = plt.subplots(len(models), len(metrics), sharex="none", sharey="none")
    fig.set_size_inches(18.5, 18.5)
    for j in range(len(metrics)):  # j is the x coordinate
        metric = metrics[j]
        for i in range(len(models)):  # i is the y coordinate
            model = models[i]
            df = data[model]
            ax[i, j].set_ylim(metrics_mapping[metric]["y_ax"])
            for prune_strat in pruning:
                df_prune = df[df["prune_method"] == prune_strat]
                x = df_prune["prune_compression"]
                y = df_prune[metric]["mean"]
                # y_err = 2* df_prune[metric]['std'] if error_bars else None #95% confidence
                y_err = (
                    df_prune[metric]["std"] if error_bars else None
                )  # 68% confidence
                if prune_strat == "GreedyPGDGlobalMagGrad":
                    ax[i, j].errorbar(
                        x, y, yerr=y_err, label=prune_mapping[prune_strat], linewidth=3
                    )
                else:
                    ax[i, j].errorbar(
                        x, y, yerr=y_err, label=prune_mapping[prune_strat]
                    )

            ax[i, j].set_xscale("log", base=2)
            ax[i, j].set_xticks(x)
            ax[i, j].xaxis.set_major_formatter(ScalarFormatter())

            if i == 0 and j == 0:
                ax[i, j].legend(prop={"size": 10}, framealpha=0.0)
            if j == 0:
                ax[i, j].set_ylabel(model_mapping[model])
            if i == 0:
                ax[i, j].set_title(metrics_mapping[metric]["label"])
            if i == len(models) - 1:
                ax[i, j].set_xlabel("Prune Compression")
            # if j==0:
            #     ax[i, j].set_ylabel(metrics_mapping[metric]['label'])

    plt.show()


if __name__ == "__main__":
    plt.style.use("ggplot")

    # plot all clean accs
    plot_all_experiment_data(
        metrics=["prune_clean_val_acc1", "quantize_clean_val_acc1"],
    )

    # plot all data with error bars
    plot_all_experiment_data_transpose(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
        error_bars=True,
    )

    # plot all data with error bars
    plot_all_experiment_data(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
        error_bars=True,
        save="all_data.png",
    )

    # plot without Layer GAP and with error bars:
    plot_all_experiment_data(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
        pruning=[
            "RandomPruning",
            "GlobalMagWeight",
            "LayerMagWeight",
            "GlobalMagGrad",
            "LayerMagGrad",
            "GreedyPGDGlobalMagGrad",
        ],
        error_bars=True,
    )

    # plot with Layer GAP and without error bars:
    plot_all_experiment_data(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "prune_attack_transfer_target_correct1",
        ],
    )

    # plot pruned robustness vs quantized robustness
    plot_all_experiment_data(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_results_adv_acc1",
            "quantize_attack_results_adv_acc1",
        ],
    )

    # plot pruned transfer vs quantized transfer
    plot_all_experiment_data(
        metrics=[
            "prune_clean_val_acc1",
            "prune_attack_transfer_target_correct1",
            "quantize_attack_transfer_target_correct1",
        ],
    )
