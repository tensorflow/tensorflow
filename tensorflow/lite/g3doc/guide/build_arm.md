# Build TensorFlow Lite for ARM boards

This page describes how to build the TensorFlow Lite libraries for ARM-based
computers.

TensorFlow Lite supports two build systems and supported features from each
build system are not identical. Check the following table to pick a proper build
system.

Feature                                                                                   | Bazel                        | CMake
----------------------------------------------------------------------------------------- | ---------------------------- | -----
Predefined toolchains                                                                     | armhf, aarch64               | armel, armhf, aarch64
Custom toolchains                                                                         | harder to use                | easy to use
[Select TF ops](https://www.tensorflow.org/lite/guide/ops_select)                         | supported                    | not supported
[GPU delegate](https://www.tensorflow.org/lite/performance/gpu)                           | only available for Android   | any platform that supports OpenCL
XNNPack                                                                                   | supported                    | supported
[Python Wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip)                     | supported                    | supported
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md) | supported                    | [supported](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
[C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)      | supported for Bazel projects | supported for CMake projects

## Cross-compilation for ARM with CMake

If you have a CMake project or if you want to use a custom toolchain, you'd
better use CMake for cross compilation. There is a separate
[Cross compilation TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake_arm)
page available for this.

## Cross-compilation for ARM with Bazel

If you have a Bazel project or if you want to use TF ops, you'd better use Bazel
build system. You'll use the integrated
[ARM GCC 8.3 toolchains](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/toolchains/embedded/arm-linux)
with Bazel to build an ARM32/64 shared library.

| Target Architecture | Bazel Configuration     | Compatible Devices         |
| ------------------- | ----------------------- | -------------------------- |
| armhf (ARM32)       | --config=elinux_armhf   | RPI3, RPI4 with 32 bit     |
:                     :                         : Raspberry Pi OS            :
| AArch64 (ARM64)     | --config=elinux_aarch64 | Coral, RPI4 with Ubuntu 64 |
:                     :                         : bit                        :

Note: The generated shared library requires glibc 2.28 or higher to run.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
and TensorFlow devel docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To cross compile TensorFlow Lite with Bazel, follow the steps:

#### Step 1. Install Bazel

Bazel is the primary build system for TensorFlow. Install the latest version of
the [Bazel build system](https://bazel.build/versions/master/docs/install.html).

**Note:** If you're using the TensorFlow Docker image, Bazel is already
available.

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already
provided in `/tensorflow_src/`.

#### Step 3. Build ARM binary

##### C library

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

You can find a shared library in:
`bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so`.

**Note:** Use `elinux_armhf` for
[32bit ARM hard float](https://wiki.debian.org/ArmHardFloatPort) build.

Check
[TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md)
page for the detail.

##### C++ library

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```

You can find a shared library in:
`bazel-bin/tensorflow/lite/libtensorflowlite.so`.

Currently, there is no straightforward way to extract all header files needed,
so you must include all header files in tensorflow/lite/ from the TensorFlow
repository. Additionally, you will need header files from FlatBuffers and
Abseil.

##### Etc

You can also build other Bazel targets with the toolchain. Here are some useful
targets.

*   //tensorflow/lite/tools/benchmark:benchmark_model
*   //tensorflow/lite/examples/label_image:label_image
