from experiments import Experiment
from utils import email_callback
from pathlib import Path
import json


def run_experiments():
    f = open(Path.cwd() / 'experiment.json')
    args = json.load(f)
    f.close()
    common_args = {x: args[x] for x in ['gpu', 'best_model_metric', 'experiment_number', 'dataset', 'debug']}

    # dummy email function in case no email is provided
    email_fn = lambda x, y: 0
    if 'email' in args:
        email_fn = email_callback(args['email'])
    for experiment_args in args['experiments']:
        experiment_args['email'] = email_fn
        experiment_args.update(common_args)

        e = Experiment(**experiment_args)
        e.run()


if __name__ == '__main__':
    run_experiments()