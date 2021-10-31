"""Run experiments based off of an experiments.json config file."""

# pylint: disable=import-error, unspecified-encoding, invalid-name

from pathlib import Path
import json
from experiments import Experiment
from utils import Email_sender


def run_experiments(filename: str = None) -> None:
    """Run the experiments described in the config file."""

    path = Path.cwd() / "experiment.json"
    if filename:
        path = Path.cwd() / filename

    with open(path, "r") as file:
        args = json.load(file)

    common_args = {
        x: args[x]
        for x in ["gpu", "best_model_metric", "experiment_number", "dataset", "debug"]
    }

    # dummy email function in case no email is provided
    email_fn = lambda x, y: 0
    if "email" in args:
        email_fn = Email_sender(**args["email"]).email
    for experiment_args in args["experiments"]:
        experiment_args["email"] = email_fn
        experiment_args.update(common_args)

        e = Experiment(**experiment_args)
        e.run()


if __name__ == "__main__":
    run_experiments()
