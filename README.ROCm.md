# Tensorflow ROCm port #

## Introduction ##

This repository hosts the port of [Tensorflow](https://github.com/tensorflow/tensorflow) on ROCm platform. It uses various technologies on ROCm platform such as HIP and MIOpen. For details on HIP, please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Optimized DNN library calls (via [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)) are also supported within this codebase.

## Installation ##

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md).

The project is derived from TensorFlow 1.13.1 and has been verified to work with the latest ROCm2.3 release.

For details on installation and usage, see these links:
* [Basic installation](rocm_docs/tensorflow-install-basic.md)
* [Building from source](rocm_docs/tensorflow-build-from-source.md)
* [Quickstart guide](rocm_docs/tensorflow-quickstart.md)


## Technical details ##
* [Overview of ROCm port](rocm_docs/rocm-port-overview.md)
* [List of supported operators on ROCm](rocm_docs/core_kernels.md)

