# Delira Baseline: Unet-Segmentation

This repository contains a U-Net implementation which is compatible with the 
[delira](https://github.com/delira-dev/delira)-Framework and its 
[PyTorch](https://pytorch.org)-Backend.

## Contents
It contains the following:
* Generic U-Net implementation for 2D and 3D
* Implementation of the SoftDiceLoss
* Implementation of the RADAM Optimizer
* Implementation of Stat-Score Calculation (TP, FP, TN, FN)
* Implementation of a Multiclass-Dice Score
* Utility Functions to convert tensors from index based formats to 
onehot encoding
* An exemplaric training script
* An exemplaric prediction script

## Usage
The easiest usage is to install it, and fill out the missing parts of the 
example scripts (essentially the data-handling).

### Installation
This repository contains an installable python-package structure which can be 
installed by pip via 
`pip install git+https://github.com/justusschock/delira-unet-baseline`


## Remaining ToDos:
- [ ] Add Unittests for CI/CD
- [ ] Add pretrained networks and predefined datasets