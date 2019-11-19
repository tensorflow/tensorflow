# Build TensorFlow Lite for ARM64 boards

This page describes how to build the TensorFlow Lite static library for
ARM64-based computers. If you just want to start using TensorFlow Lite to
execute your models, the fastest option is to install the TensorFlow Lite
runtime package as shown in the [Python quickstart](python.md).

Note: This page shows how to compile only the C++ static library for
TensorFlow Lite. Alternative install options include: [install just the Python
interpreter API](python.md) (for inferencing only); [install the full
TensorFlow package from pip](https://www.tensorflow.org/install/pip);
or [build the full TensorFlow package](
https://www.tensorflow.org/install/source).

## Cross-compile for ARM64

To ensure the proper build environment, we recommend using one of our TensorFlow
Docker images such as [tensorflow/tensorflow:nightly-devel](
https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To get started, install the toolchain and libs:

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

If you are using Docker, you may not use `sudo`.

Now git-clone the TensorFlow repository
(`https://github.com/tensorflow/tensorflow`)—if you're using the TensorFlow
Docker image, the repo is already provided in `/tensorflow_src/`—and then run
this script at the root of the TensorFlow repository to download all the
build dependencies:

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

Note that you only need to do this once.

Then compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.

## Compile natively on ARM64

These steps were tested on HardKernel Odroid C2, gcc version 5.4.0.

Log in to your board and install the toolchain:

```bash
sudo apt-get install build-essential
```

Now git-clone the TensorFlow repository
(`https://github.com/tensorflow/tensorflow`) and run this at the root of
the repository:

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

Note that you only need to do this once.

Then compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.
