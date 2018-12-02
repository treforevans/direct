# Discrete Relaxation of Continuous Variables for tractable Variational Inference (DIRECT)
This repository contains a Tensorflow implementation to perform variational inference by discretely relaxing continuous variables (DIRECT).
[The paper](https://arxiv.org/abs/1809.04279) outlining this method by Trefor W. Evans and Prasanth B. Nair appears at NIPS 2018.

Please cite our paper if you find this code useful in your research. The bibliographic information for the paper is
```
@inproceedings{evans_direct,
  title={Discrete Relaxation of Continuous Variables for tractable Variational Inference,
  author={Evans, Trefor W and Nair, Prasanth B},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

## Overview
*Checkout the [3-minute video](https://youtu.be/x0XzyEJY0ds) outlining the central ideas in [our paper](https://arxiv.org/abs/1809.04279).*

In the DIRECT approach to variational inference, we discretely relax continuous variables such that posterior samples consist of sparse and low-precision quantized integers.
This enables memory and energy efficient inference which is critical for on-board machine learning on mobile devices as well as large-scale deployed models.
Variational inference for discrete latent variable models typically require the use of high variance stochastic gradient estimators, making training impractical for large-scale models.
Instead, the DIRECT approach exploits algebraic structure of the ELBO, enabling
* exact computation of ELBO gradients (i.e. unbiased, zero-variance gradient estimates) for fast convergence,
* a training complexity that's *independent* of the number of training points, permitting inference on large datasets, and
* exact computation of statistical moments of the predictive posterior without relying on Monte Carlo sampling.

The DIRECT approach is not practical for all likelihoods, however, [our paper](https://arxiv.org/abs/1809.04279) identifies a couple of popular models that are practical,
and demonstrates efficient inference on huge datasets with an extremely low-precision (4-bit) quantized integer relaxation.
This repository contains a DIRECT generalized linear model that can be trained on massive datesets with either a factorized (mean-field) variational distribution, or an unfactorized mixture distribution.

## Usage
This package contains just one single class, `direct.BayesGLM`, which is a DIRECT generalized linear model and is implemented in only ~300 lines of code.
To see how to use this class, see the [tutorial notebook](/tutorial.ipynb) which demonstrates usage with both a factorized (mean-field) variational distribution, and an unfactorized mixture variational distribution.

## Dependencies
This code has the following dependencies (version number crucial):
* python==3.6
* tensorflow==1.10

You can create a conda environment and set it up using the provided `environment.yaml` file, as follows
```
conda env create -f environment.yaml -n direct
```
then activate the environment using
```
source activate direct
```
