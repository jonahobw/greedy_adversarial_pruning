# Greedy Adversarial Pruning

Official repo for the paper

[Hardening DNNs against Transfer Attacks during Network Compression using Greedy Adversarial Pruning](https://doi.org/10.1109/AICAS54282.2022.9869910)

Authors: [Jonah O'Brien Weiss](https://jonahobw.github.io[), [Tiago Alves](https://scholar.google.com/citations?user=8MmE3TUAAAAJ&hl=en), and [Sandip Kundu](https://people.umass.edu/kundu/)

Published in the 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS)

---

## Getting Started

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/)
- Python 3.9

### Installation
Run `inshall.sh`.  This will create a conda environment called "gap", install the requirements, and setup the submodule dependency on the "shrinkbench" repo.

### Running Experiments
1. Modify [src/config.yaml](src/config.yaml) to define what experiments to run.  This is prepopulated with an example configuration in debug mode.  Configs get saved in a new file when running experiements.
2. Run `download_dataset.py`
3. Run `run.py`.

---

## Folder Structure

Notable files under [src](src):

```
src/
├── analyze_data.py           # Scripts for plotting experiment results and data
├── config.yaml               # Main configuration file for experiments
├── download_dataset.py       # Script to download required datasets
├── email_pw.txt              # (Sensitive) Email password for notifications (should not be shared)
├── evaulate_model.py         # Model evaluation utilities and scripts
├── experiments.py            # Main experiment orchestration logic
├── remove_models.py          # Script to remove or clean up models
├── run.py                    # Entry point to run experiments
└── shrinkbench/              # Submodule: ShrinkBench library for pruning and compression
    ├── experiment/          # Core experiment tracking and logging functionality
    ├── models/              # Neural network model definitions and utilities
    ├── pruning/             # Base pruning classes and adversarial pruning implementations
    └── strategies/          # Different pruning strategy implementations (magnitude, gradient, etc.)
```

---

## Contact

_For questions or issues, please contact [jgow98@gmail.com](mailto:jgow98@gmail.com)._ 