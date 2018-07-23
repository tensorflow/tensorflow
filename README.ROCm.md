# Tensorflow ROCm port #

## Introduction ##

This repository hosts the port of [Tensorflow](https://github.com/tensorflow/tensorflow) on ROCm platform. It uses various technologies on ROCm platform such as HIP and MIOpen. For details on HIP, please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Optimized DNN library calls (via [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)) are also supported within this codebase.

## Installation ##

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md).

The project is derived from TensorFlow 1.8.0 and has been verified to work with the latest ROCm 1.8.2 release.

For details on installation and usage, see these links:
* [Basic installation](rocm_docs/tensorflow-install-basic.md)
* [Building from source](rocm_docs/tensorflow-build-from-source.md)
* [Quickstart guide](rocm_docs/tensorflow-quickstart.md)


## Technical details ##
* [Overview of ROCm port](rocm_docs/rocm-port-overview.md)
* [List of supported operators on ROCm](rocm_docs/core_kernels.md)


## Known Issues / Workarounds

### tensorflow/benchmarks workaround for TF v1.8
Since our current port of TF supports v1.8, we can't use some of the newest commits in `tensorflow/benchmarks`. RCCL, a ROCm version of NCCL, is under implementation. Therefore we have to drop back to an earlier point in the commit history, and disable NCCL.

```
git checkout -b may22 ddb23306fdc60fefe620e6ce633bcd645561cb0d && \
sed -i 's|from tensorflow.contrib import nccl|#from tensorflow.contrib import nccl|g' ./scripts/tf_cnn_benchmarks/variable_mgr.py
```
