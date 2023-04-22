# Building TensorFlow Lite Standalone Pip

Many users would like to deploy TensorFlow lite interpreter and use it from
Python without requiring the rest of TensorFlow.

## Steps

To build a binary wheel run this script:

```sh
sudo apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
pip install numpy pybind11
sh tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh
```

That will print out some output and a .whl file. You can then install that

```sh
pip install --upgrade <wheel>
```

You can also build a wheel inside docker container using make tool. For example
the following command will cross-compile tflite-runtime package for python2.7
and python3.7 (from Debian Buster) on Raspberry Pi:

```sh
make BASE_IMAGE=debian:buster PYTHON=python TENSORFLOW_TARGET=rpi docker-build
make BASE_IMAGE=debian:buster PYTHON=python3 TENSORFLOW_TARGET=rpi docker-build
```

Another option is to cross-compile for python3.5 (from Debian Stretch) on ARM64
board:

```sh
make BASE_IMAGE=debian:stretch PYTHON=python3 TENSORFLOW_TARGET=aarch64 docker-build
```

To build for python3.6 (from Ubuntu 18.04) on x86_64 (native to the docker
image) run:

```sh
make BASE_IMAGE=ubuntu:18.04 PYTHON=python3 TENSORFLOW_TARGET=native docker-build
```

In addition to the wheel there is a way to build Debian package by adding
BUILD_DEB=y to the make command (only for python3):

```sh
make BASE_IMAGE=debian:buster PYTHON=python3 TENSORFLOW_TARGET=rpi BUILD_DEB=y docker-build
```

## Alternative build with Bazel (experimental)

There is another build steps to build a binary wheel which uses Bazel instead of
Makefile. You don't need to install additional dependencies.
This approach can leverage TF's ci_build.sh for ARM cross builds.

### Normal build for your workstation

```sh
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh
```

### Optimized build for your workstation
The output may have a compatibility issue with other machines but it gives the
best performance for your workstation.

```sh
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh native
```

### Cross build for armhf Python 3.5

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf
```

### Cross build for armhf Python 3.7

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf
```

### Cross build for aarch64 Python 3.5

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64
```

### Cross build for aarch64 Python 3.8

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON38 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64
```

### Cross build for aarch64 Python 3.9

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON39 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64
```

### Native build for Windows

```sh
bash tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh windows
```

## Enable TF OP support (Flex delegate)

If you want to use TF ops with Python API, you need to enable flex support.
You can build TFLite interpreter with flex ops support by providing
"--define=tflite_pip_with_flex=true" to Bazel.

Here are some examples.

### Normal build with Flex for your workstation

```sh
CUSTOM_BAZEL_FLAGS=--define=tflite_pip_with_flex=true \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh
```

### Cross build with Flex for armhf Python 3.7

```sh
CI_DOCKER_EXTRA_PARAMS="-e CUSTOM_BAZEL_FLAGS=--define=tflite_pip_with_flex=true" \
  tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf
```

## Usage

Note, unlike tensorflow this will be installed to a tflite_runtime namespace.
You can then use the Tensorflow Lite interpreter as.

```python
from tflite_runtime.interpreter import Interpreter
interpreter = Interpreter(model_path="foo.tflite")
```

This currently works to build on Linux machines including Raspberry Pi. In
the future, cross compilation to smaller SOCs like Raspberry Pi from
bigger host will be supported.

## Caveats

* You cannot use TensorFlow Select ops, only TensorFlow Lite builtins.
* Currently custom ops and delegates cannot be registered.

