# Build TensorFlow Lite for ARM64 boards

This page describes how to build the TensorFlow Lite static and shared libraries
for ARM64-based computers. If you just want to start using TensorFlow Lite to
execute your models, the fastest option is to install the TensorFlow Lite
runtime package as shown in the [Python quickstart](python.md).

Note: This page shows how to compile only the C++ static and shared libraries
for TensorFlow Lite. Alternative install options include:
[install just the Python interpreter API](python.md) (for inferencing only);
[install the full TensorFlow package from pip](https://www.tensorflow.org/install/pip);
or
[build the full TensorFlow package](https://www.tensorflow.org/install/source).

**Note:** Cross-compile ARM with CMake is available. Please check
[this](https://www.tensorflow.org/lite/guide/build_cmake_arm).

## Cross-compile for ARM64 with Make

To ensure the proper build environment, we recommend using one of our TensorFlow
Docker images such as
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To get started, install the toolchain and libs:

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

If you are using Docker, you may not use `sudo`.

Now git-clone the TensorFlow repository
(https://github.com/tensorflow/tensorflow)—if you're using the TensorFlow Docker
image, the repo is already provided in `/tensorflow_src/`—and then run this
script at the root of the TensorFlow repository to download all the build
dependencies:

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

Note that you only need to do this once.

Then compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`.

## Compile natively on ARM64

These steps were tested on HardKernel Odroid C2, gcc version 5.4.0.

Log in to your board and install the toolchain:

```bash
sudo apt-get install build-essential
```

Now git-clone the TensorFlow repository
(https://github.com/tensorflow/tensorflow) and run this at the root of the
repository:

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

Note that you only need to do this once.

Then compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a`.

## Cross-compile for ARM64 with Bazel

You can use
[ARM GCC toolchains](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)
with Bazel to build an ARM64 shared library.

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

#### Step 3. Build ARM64 binary

##### C library

```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

Check
[TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)
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
