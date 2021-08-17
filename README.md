# STENCIL-NET

This repository contains the code for the discretization of PDEs from data using STENCIL-NET and is accompanying the 
paper "STENCIL-NET: Data-driven solution-adaptive discretization of partial differential equations" submitted to 
NeurIPS 2020. All the codes have been implemented using PyTorch and are properly documented.

## Dependencies

1. STENCIL-NET
    - PyTorch (1.5.0)
    - NumPy
    - SciPy
    
2. Example Notebooks
    - Jupyter
    - Matplotlib
    - tqdm

## Contents of Repository

- `stencilnet` module for neural network setup and time integrators
- `utils.py` and the `data` folder contain all the neccessary codes and data to reproduce the denoising and simulation 
  of KdV, forced Burgers and KS equations
    - forced Burgers equation: `utils.burgers_simulation`
    - KdV equation: `data/kdv_raissi.mat`
    - KS equation: `data/KS_x256_t1000.mat`

- The notebooks show a few examples on how to use STENCIL-NET for the tasks at hand
- Pre-trained models for `KdVNoiseDecomposition` and `ForcedBurgersSimulation` which can be loaded with the convenience 
  functions `utils.load_denoising_model` and `utils.load_simulation_model`. The exact hyperparameters used for training 
  are visible in the corresponding file names.