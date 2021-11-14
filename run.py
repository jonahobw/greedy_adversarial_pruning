"""Run experiments based off of an experiments.json config file."""

# pylint: disable=import-error, unspecified-encoding, invalid-name

from pathlib import Path
import yaml
import time
import traceback
import copy
import logging

from shrinkbench.util import OnlineStats

from experiments import Experiment
from experiment_utils import Email_Sender, timer, generate_permutations

logger = logging.getLogger('run')

def run_experiments(filename: str = None) -> None:
    """Run the experiments described in the config file."""

    path = Path.cwd() / "config.yaml"
    if filename:
        path = Path.cwd() / filename

    logger.info(f"Reading config file {path}")

    with open(path, "r") as file:
        args = yaml.safe_load(file)

    common_args = args['common_args']

    # this variable gets passed to Experiment class, it is overwritten to a valid email sender
    # if valid email credentials are given in the config file.
    email_fn = None

    # whether or not to send emails once per experiment (done from this script) or
    # more frequently (done from the Experiment class).  If this is present in the config
    # file, then this variable is overwritten.
    once_per_experiment = False

    if "email" in args:
        email_fn = Email_Sender(**args["email"]).email
        if "once_per_experiment" in args["email"]:
            once_per_experiment = args["email"]["once_per_experiment"]

    num_experiments = len(args["experiments"])
    experiment_time = OnlineStats()

    now = time.time()

    for i, experiment_args in enumerate(args["experiments"]):
        try:
            since = now
            if not once_per_experiment and email_fn is not None:
                experiment_args["email"] = email_fn

            experiment_args.update(copy.deepcopy(common_args))

            e = Experiment(**experiment_args)
            e.run()

            now = time.time()
            # we just completed this experiment, hence the - 1 at the end
            left = num_experiments - i - 1
            done_percent = "{:.0f}".format((i + 1) / num_experiments * 100)
            last_exp_time = now - since
            experiment_time.add(last_exp_time)
            estimated_time_remaining = timer(left * experiment_time.mean)

            # handle email sending from this script rather than the Experiment class
            if email_fn is not None and once_per_experiment:
                email_fn(
                    f"Experiment completed for {e.name}, "
                    f"{left} Experiments Left, {done_percent}% Completed",
                    f"Time of last experiment: {timer(now-since)}\n"
                    f"Estimated time remaining ({left} experiments left and "
                    f"{timer(experiment_time.mean)} per experiment): "
                    f"{estimated_time_remaining}\n\n"
                    f"{e.params}\n")

        except Exception as e:
            tb = traceback.format_exc()
            if email_fn is not None:
                email_fn("PROGRAM CRASHED", f"{tb}")
            raise e

    if email_fn is not None:
        email_fn(
            "All Experiments Concluded.",
            f"Total time ({num_experiments} experiments @ {timer(experiment_time.mean)} per "
            f"experiment): {timer(num_experiments * experiment_time.mean)}"
        )

def convert_experiment_list(args):
    """
    Convert the experiment arguments passed from the config file to a list.

    The config file allows for multiple experiments to be run by specifying a list of values for
    any experiment parameter.  For example, this function will convert a config file with
    experiment parameters as

    -   model_type: [vgg_bn_drop, resnet20]
        prune_method: [RandomPruning, GlobalMagWeight]
        prune_compression: 2
        finetune_epochs: 40

    to a list of experiment parameters as

    [
        {
            "model_type": "vgg_bn_drop",
            "prune_method": "RandomPruning",
            "prune_compression": 2
            "finetune_epochs": 40
        },
        {
            "model_type": "vgg_bn_drop",
            "prune_method": "GlobalMagWeight",
            "prune_compression": 2
            "finetune_epochs": 40
        },
        {
            "model_type": "resnet20",
            "prune_method": "RandomPruning",
            "prune_compression": 2
            "finetune_epochs": 40
        },
        {
            "model_type": "resnet20",
            "prune_method": "GlobalMagWeight",
            "prune_compression": 2
            "finetune_epochs": 40
        }
    ]

    :param args: the dictionary resulting from parsing the config file.

    :returns: a list of dictionaries, each of which represents the kwargs for 1 Experiment object.
    """

    # dict of args which have a list
    list_args = {}

    # dict of args that do not have a list
    common_args = {}

    for arg in args["experiments"]:
        if isinstance(args["experiments"][arg], list):
            arr[arg] = args["experiments"][arg]
        else:
            common_args[arg] = args["experiments"][arg]

    # now create all permutations of the list_args:
    permutations = generate_permutations(list_args)

    for permutation in permutations:
        permutation.update(common_args)

    return permutations

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-file', required=False,
                        default='config.yaml', help='.yaml file with configuration')

    args = parser.parse_args()
    run_experiments(args.file)
