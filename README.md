# dynamic_stability_datasets_gnn_paper-companion

This repository contains the code to reproduce the results of the paper: "Toward dynamic stability assessment of power grid topologies using graph neural networks" with the DOI: https://doi.org/10.1063/5.0160915. The repository contains the code to generate the datasets published on Zenodo: https://zenodo.org/record/6572973 as well as the code to train models of the corresponding paper. The datasets are generated using Julia and for the training of the ML models Pytorch is used. More information to reproduce figures can be found on Zenodo: https://doi.org/10.5281/zenodo.8204334


## Generation of the datasets
The generation of the datasets occurs in multiple steps. To reproduce the results, we recommend using Julia 1.5.3 and the provided Project.toml and Manifest.toml to have the same environment. The corresponding path to activate the environment has to be set in all files. After manipulation of the paths, the scripts can be consecutively executed. Keep in mind that the dynamical computations are very expensive (~550,000 CPU hours for dataset20 and dataset100).

1. Grid generation: generate_grids_and_seeds.jl
2. Dynamical computation: compute_dynamics.jl
3. Preparation for  ML training: prepare4Pytorch.jl

Prior to executing the scripts, the paths to store/loading the grid data and store/loading the dynamical results, as well as the number of desired nodes per grid must be adapted in generate_grids_and_seeds.jl and compute_dynamics.jl. Furthermore, the path to the girds, dynamical results and the output directory for the ML data have to be set in prepare4Pytorch.jl.

## Training of the ML models
To evaluate the reproducibility, we provide one Jupyter notebook and a python script to train a small TAG model on the first 1,000 grids after automatically downloading the full datasets from Zenodo. We provide a conda environment file to generate a conda environment including all the necessary software. 
To create the conda environment, the ```ENVNAME``` can be chosen freely. 
```
conda env create -n ENVNAME --file conda_environment.yml
```
The file conda_environment.yml is stored in training_model.
Afterwards the script train_small_TAG.py or train_small_TAG.ipynb can be executed.


In case of any problems or questions, do not hesitate to contact us.
