# Build TensorFlow Lite Python Wheel Package

This page describes how to build the TensorFlow Lite `tflite_runtime` Python
library for x86_64 and various ARM devices.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, macOS Catalina (x86_64) and TensorFlow devel Docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This feature is available since version 2.4.

#### Prerequisites

You need CMake installed and a copy of the TensorFlow source code. Please check
[Build TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake)
page for the details.

To build the PIP package for your workstation, you can run the following
commands.

```sh
PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

**Note:** If you have multiple Python interpreters available, specify the exact
Python version with `PYTHON` variable. (Currently, it supports Python 3.7 or
higher)

## ARM cross compilation

For ARM cross compilation, it's recommended to use Docker since it makes easier
to setup cross build environment. Also you needs a `target` option to figure out
the target architecture.

There is a helper tool in Makefile `tensorflow/lite/tools/pip_package/Makefile`
available to invoke a build command using a pre-defined Docker container. On a
Docker host machine, you can run a build command as followings.

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=<target> PYTHON_VERSION=<python3 version>
```

**Note:** Python version 3.7 or higher is supported.

### Available target names

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` script needs
a target name to figure out target architecture. Here is the list of supported
targets.

Target    | Target architecture  | Comments
--------- | -------------------- | --------
armhf     | ARMv7 VFP with Neon  | Compatible with Raspberry Pi 3 and 4
rpi0      | ARMv6                | Compatible with Raspberry Pi Zero
aarch64   | aarch64 (ARM 64-bit) | [Coral Mendel Linux 4.0](https://coral.ai/) <br/> Raspberry Pi with [Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi)
native    | Your workstation     | It builds with "-mnative" optimization
<default> | Your workstation     | Default target

### Build examples

Here are some example commands you can use.

#### armhf target for Python 3.7

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=armhf PYTHON_VERSION=3.7
```

#### aarch64 target for Python 3.8

```sh
make -C tensorflow/lite/tools/pip_package docker-build \
  TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
```

#### How to use a custom toolchain?

If the generated binaries are not compatible with your target, you need to use
your own toolchain or provide custom build flags. (Check
[this](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment)
to understand your target environment) In that case, you need to modify
`tensorflow/lite/tools/cmake/download_toolchains.sh` to use your own toolchain.
The toolchain script defines the following two variables for the
`build_pip_package_with_cmake.sh` script.

Variable       | Purpose                  | example
-------------- | ------------------------ | -------------------------------
`ARMCC_PREFIX` | defines toolchain prefix | arm-linux-gnueabihf-
`ARMCC_FLAGS`  | compilation flags        | -march=armv7-a -mfpu=neon-vfpv4

**Note:** `ARMCC_FLAGS` might need to contain Python library include path. See
the `download_toolchains.sh` for the reference.
