# TensorFlow Lite for generic ARM64 boards

## Cross compiling

### Installing the toolchain

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

> If you are using Docker, you may not use `sudo`.

### Building

Clone this Tensorflow repository.
Run this script at the root of the repository to download all the dependencies:

> The Tensorflow repository is in `/tensorflow` if you are using `tensorflow/tensorflow:nightly-devel` docker image, just try it.

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

Note that you only need to do this once.

Compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/gen/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.

## Native compiling

These steps were tested on HardKernel Odroid C2, gcc version 5.4.0.

Log in to your board, install the toolchain.

```bash
sudo apt-get install build-essential
```

First, clone the TensorFlow repository. Run this at the root of the repository:

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```
Note that you only need to do this once.

Compile:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

This should compile a static library in:
`tensorflow/lite/gen/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.
