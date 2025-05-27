#! /bin/bash

# create conda environment
conda create -n gap python=3.9
conda activate gap

# install requirements
pip install -r requirements.txt

# install shrinkbench and reactivate environment
git submodule update --init --recursive
conda env config vars set PYTHONPATH="src/shrinkbench"
conda activate gap