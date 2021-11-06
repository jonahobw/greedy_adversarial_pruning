"""Run experiments based off of an experiments.json config file."""

# pylint: disable=import-error, unspecified-encoding, invalid-name

from pathlib import Path
import yaml
import time

from shrinkbench.util import OnlineStats

from experiments import Experiment
from experiment_utils import Email_Sender, timer


def run_experiments(filename: str = None) -> None:
    """Run the experiments described in the config file."""

    path = Path.cwd() / "config.yaml"
    if filename:
        path = Path.cwd() / filename

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

            experiment_args.update(common_args)

            e = Experiment(**experiment_args)

            time.sleep(10)
            now = time.time()
            # handle email sending from this script rather than the Experiment class
            if email_fn is not None and once_per_experiment:
                # we just completed this experiment, hence the - 1 at the end
                left = num_experiments - i - 1
                done_percent = "{:.0f}".format((i + 1)/num_experiments * 100)
                last_exp_time = now-since
                experiment_time.add(last_exp_time)
                estimated_time_remaining = timer(left * experiment_time.mean)
                email_fn(
                    f"Starting Experiment for {e.name}, "
                    f"{left} Experiments Left, {done_percent}% Completed",
                    f"Time of last experiment: {timer(now-since)}\n"
                    f"Estimated time remaining ({left} experiments left and "
                    f"{experiment_time.mean} per experiment): {estimated_time_remaining}\n\n"
                    f"{e.params}\n")

        except Exception as e:
            if email_fn is not None:
                email_fn("PROGRAM CRASHED", f"{str(e)}")
            raise e

        if email_fn is not None:
            email_fn(
                "All Experiments Concluded.",
                f"Total time ({num_experiments} @ {experiment_time.mean} per "
                f"experiment): {timer(num_experiments * experiment_time.mean)}"
            )


if __name__ == "__main__":
    run_experiments()
