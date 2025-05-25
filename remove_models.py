from pathlib import Path

"""
Scripts for cleaning up model checkpoints and reporting storage usage in experiments.

Includes utilities to remove, list, and summarize storage of model files.
"""


def remove_models(path=None):
    if path is None:
        path = Path.cwd() / "experiments"

    models = list(path.glob("**/*.pt"))
    num_models = len(models)
    if num_models == 0:
        print("All models deleted.")
        return

    storage = 0.0

    for model in models:
        # print(model)
        storage += model.stat().st_size * 0.000001  # MB
        model.unlink()

    print(f"Removed {num_models} models, cleared {storage} MB.")


def list_models(root=None):
    if not root:
        root = Path.cwd() / "experiments" / f"experiment_0"

    models = ["mobilenet_v2", "resnet20", "googlenet", "vgg_bn_drop"]

    paths = [root / x / "CIFAR10" for x in models]

    storage = 0.0

    for path in paths:
        model_files = list(path.glob("**/*.pt"))
        model_files = [str(x.relative_to(Path.cwd())) for x in model_files]
        print(model_files)
        for file in model_files:
            storage += Path(file).stat().st_size * 0.000001  # MB
            print("'" + file + "',")
        print("\n\n\n")

    print(f"Total model storage: {storage} MB")
    return storage


def list_all_models():
    exp_folder = Path.cwd() / "experiments"
    paths = list(exp_folder.glob("experiment_*"))
    all_storage = 0
    for path in paths:
        print(path)
        all_storage += list_models(path)

    print(f"Storage of all models and all experiments is {all_storage}MB.")


if __name__ == "__main__":
    ans = input("Delete models? (Answer 'y' to delete, 'n' to list models)\n")
    if ans == "y":
        remove_models()
    if ans == "n":
        list_all_models()
