# Tensorflow ROCm port #

## Introduction ##

This repository hosts the port of [Tensorflow](https://github.com/tensorflow/tensorflow) on ROCm platform. It uses various technologies on ROCm platform such as HIP and MIOpen. For details on HIP, please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Optimized DNN library calls (via [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)) are also supported within this codebase.

## Installation ##

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md).

The project is derived from TensorFlow 1.3.0 and has been verified to work with the latest ROCm 1.7.1 release.

For details on installation and usage, see these links:
* [Basic installation](rocm_docs/tensorflow-install-basic.md)
* [Building from source](rocm_docs/tensorflow-build-from-source.md)
* [Quickstart guide](rocm_docs/tensorflow-quickstart.md)


## Technical details ##
* [Overview of ROCm port](rocm_docs/rocm-port-overview.md)
* [List of supported operators on ROCm](rocm_docs/core_kernels.md)


## Known Issues / Workarounds

### X freezes under load
ROCm 1.7.1 a kernel parameter `noretry` has been set to 1 to improve overall system performance. However it has been proven to bring instability to graphics driver shipped with Ubuntu. This is an ongoing issue and we are looking into it.

Before that, please try apply this change by changing `noretry` bit to 0.

```
echo 0 | sudo tee /sys/module/amdkfd/parameters/noretry
```

Files under `/sys` won't be preserved after reboot so you'll need to do it every time.

One way to keep `noretry=0` is to change `/etc/modprobe.d/amdkfd.conf` and make it be:

```
options amdkfd noretry=0
```

Once it's done, run `sudo update-initramfs -u`. Reboot and verify `/sys/module/amdkfd/parameters/noretry` stays as 0.

For more information please check this [issue report](https://github.com/ROCmSoftwarePlatform/tensorflow/issues/13)

### tensorflow/benchmarks workaround for TF v1.3
Since our current port of TF supports v1.3, we can't use some of the newest commits in `tensorflow/benchmarks`.  One specific error you could observe is a `ImportError: cannot import name batching`.  "Batching" was introduced in this [commit](https://github.com/tensorflow/benchmarks/commit/82dd0539c76afa8491e50d8f796e686b4d97b988). Also RCCL, a ROCm version of NCCL, is under implementation. Therefore we have to drop back to an earlier point in the commit history, and disable NCCL.

```
git checkout -b sep7 6a33b4a4b5bda950bb7e45faf13120115cbfdb2f
sed -i 's|from tensorflow.contrib import nccl|#from tensorflow.contrib import nccl|g' ./scripts/tf_cnn_benchmarks/variable_mgr.py
```
