# Build from source

This document describes how to build XLA components.

If you did not clone the XLA repository or install Bazel, please check out the
"Get started" section of the README document.

## Linux

### Configure

XLA builds are configured by the `.bazelrc` file in the repository's root
directory. The `./configure` or `./configure.py` scripts can be used to adjust
common settings.

If you need to change the configuration, run the `./configure` script from the
repository's root directory. This script will prompt you for the location of XLA
dependencies and asks for additional build configuration options (compiler
flags, for example). Refer to the *Sample session* section for details.

```
./configure
```

There is also a python version of this script, `./configure.py`. If using a
virtual environment, `python configure.py` prioritizes paths within the
environment, whereas `./configure` prioritizes paths outside the environment. In
both cases you can change the default.

### CPU support

We recommend using a suitable docker container to build/test XLA, such as
[TensorFlow's docker container](https://www.tensorflow.org/install/docker):

```
docker run --name xla -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
```

Using a docker container you can build XLA with CPU support using the following commands:

```
docker exec xla ./configure
docker exec xla bazel build //xla/...  --spawn_strategy=sandboxed --test_output=all
```

If you want to build XLA targets with CPU support without Docker you need to install gcc-10:

```
apt install gcc-10 g++-10
```

Then configure and build targets using the following commands:
```
yes '' | GCC_HOST_COMPILER_PATH=/usr/bin/gcc-10 CC=/usr/bin/gcc-10 TF_NEED_ROCM=0 TF_NEED_CUDA=0 TF_CUDA_CLANG=0 ./configure

bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```


### GPU support

We recommend using a GPU docker container to build XLA with GPU support, such
as:

```
docker run --name xla_gpu -w /xla -it -d --rm -v $PWD:/xla tensorflow/tensorflow:devel-gpu bash
```

To build XLA with GPU support use the following command:

```
docker exec -e TF_NEED_CUDA=1 xla_gpu ./configure
docker exec xla_gpu bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

If you want to build XLA targets with GPU support without Docker you need to install the following dependencies additional to CPU dependencies: [`cuda-11.2`](https://developer.nvidia.com/cuda-11.2.2-download-archive), [`cuDNN-8.1`](https://developer.nvidia.com/cudnn).

Then configure and build targets using the following commands:

```
yes '' | GCC_HOST_COMPILER_PATH=/usr/bin/gcc-10 CC=/usr/bin/gcc-10 TF_NEED_ROCM=0 TF_NEED_CUDA=1 TF_CUDA_CLANG=0 ./configure

bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```


For more details regarding
[TensorFlow's GPU docker images you can check out this document.](https://www.tensorflow.org/install/source#gpu_support_3)
