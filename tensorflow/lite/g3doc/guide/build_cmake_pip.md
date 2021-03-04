# Build TensorFlow Lite Python Wheel Package

This page describes how to build the TensorFlow Lite `tflite_runtime` Python
library for x86_64 and various ARM devices.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, TensorFlow devel Docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This feature is currently experimental and available since version 2.4
and may change.

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
Python version with `PYTHON` variable. (Currently, it supports Python 3.5 or
higher)

## ARM cross compilation

For ARM cross compilation, it's recommanded to use Docker since it makes easier
to setup cross build environment. Also you needs a `target` option to figure out
the target architecture.

With the `container` name and the `target` name, you can run the build command
as followings.

```sh
tensorflow/tools/ci_build/ci_build.sh <container> \
  tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh <target>
```

### Available Docker containers

You need to select ARM cross build container for your target Python interpreter
version. Here is the list of supported containers.

Conainter   | Supported Python version
----------- | ------------------------
PI          | Python 3.5
PI-PYTHON37 | Python 3.7
PI-PYTHON38 | Python 3.8

### Available target names

`tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh` script needs
a target name to figure out target architecture. Here is the list of supported
targets.

Target    | Target architecture  | Comments
--------- | -------------------- | --------
armhf     | ARMv7 VFP with Neon  | Compatibile with Raspberry Pi 3 and 4
rpi0      | ARMv6                | Compatibile with Raspberry Pi Zero
aarch64   | aarch64 (ARM 64-bit) | [Coral Mendel Linux 4.0](https://coral.ai/) <br/> Raspberry Pi with [Ubuntu Server 20.04.01 LTS 64-bit](https://ubuntu.com/download/raspberry-pi)
native    | Your workstation     | It builds with "-mnative" optimization
<default> | Your workstation     | Default target

### Build examples

Here are some example commands you can use.

#### armhf target for Python 3.7

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh armhf
```

#### aarch64 target for Python 3.8

```sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON38 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh aarch64
```

#### How to use a custom toolchain?

If the generated binaries are not compatibile with your target, you need to use
your own toolchain or provide custom build flags. (Check
[this](https://www.tensorflow.org/lite/guide/build_cmake_arm#check_your_target_environment)
to understand your target environment) In that case, you need to modify
`tensorflow/lite/tools/cmake/download_toolchains.sh` to use your own toolchain.
The toolchain script defines the following two variables for the
`build_pip_package_with_cmake.sh` script.

Variable     | Purpose                  | example
------------ | ------------------------ | -------------------------------
ARMCC_PREFIX | defines toolchain prefix | arm-linux-gnueabihf-
ARMCC_FLAGS  | compilation flags        | -march=armv7-a -mfpu=neon-vfpv4

**Note:** ARMCC_FLAGS might need to contain Python library include path. See the
`download_toolchains.sh` for the reference.
