# Loss Landscape Geometry & Optimization Dynamics  
### A Theoretical + Empirical Investigation using PyTorch

This repository contains my full submission for the **Round 2 Assignment** of the FourKites hiring process.  
The goal of this project is to develop a rigorous framework for analyzing the **geometry of neural network loss landscapes**,  
and to connect geometric properties with **optimization behavior**, **generalization**, and **model robustness**.

The project includes:
- A **detailed LaTeX report** (with theory + experiments + plots)
- Complete **PyTorch implementation** for all experiments
- **Training, loss slicing, Hessian eigenvalue estimation, mode connectivity**, and **robustness analysis**
- All generated visualizations and supporting materials


#  Project Overview

Modern deep neural networks exhibit surprising generalization ability even when optimized on highly non-convex loss functions.  
This project studies:

###  Why SGD finds generalizable minima  
###  How landscape geometry influences optimization  
###  How curvature (Hessian), sharpness, and flatness relate to generalization  
###  Whether independently trained minima are connected  
###  How robust a solution is to perturbations in parameter space  

To answer these questions, we conduct **carefully designed experiments** on Fashion-MNIST using PyTorch, analyzing:

- Training trajectories  
- Top Hessian eigenvalues  
- 1D and 2D loss landscape slices  
- Mode connectivity between minima  
- Sensitivity to random weight perturbations  

# Repository Structure

```
.
├── report/
│ ├── report.pdf
├── src/
│ ├── model.py # CNN model definition
│ ├── train.py # Training/evaluation code
│ ├── hessian.py # Hessian-vector product + power iteration
│ ├── landscape.py # 1D and 2D loss landscape slicing
│ ├── connectivity.py # Mode connectivity analysis
│ ├── robustness.py # Weight perturbation experiments
│ └── utils.py # Flattening helpers, common utilities
│
├── notebooks/
│ ├── experiments.ipynb # Notebook version of full pipeline
└── README.md
```


This structure follows standard ML research practices for clarity and reproducibility.

---

# Setup Instructions

### 1️ Clone the repository

```
bash
git clone https://github.com/shubham171002/FourKites_Assessment.git
cd FourKites_Assessment
```

### 2️ Create environment (recommended: conda)

```
conda create -n landscape python=3.10 -y
conda activate landscape
```

### 3️ Install dependencies

```
torch
torchvision
matplotlib
numpy
```

### 4 Running the Project
```
jupyter notebook notebooks/experiments.ipynb
```
