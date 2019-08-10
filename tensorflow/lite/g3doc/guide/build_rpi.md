# Build TensorFlow Lite for Raspberry Pi

This page describes how to build the TensorFlow Lite static library for
Raspberry Pi. If you just want to start using TensorFlow Lite to execute your
models, the fastest option is to install the TensorFlow Lite runtime package as
shown in the [Python quickstart](python.md).

Note: This page shows how to compile only the C++ static library for
TensorFlow Lite. Alternative install options include: [install just the Python
interpreter API](python.md) (for inferencing only); [install the full
TensorFlow package from pip](https://www.tensorflow.org/install/pip);
or [build the full TensorFlow package](
https://www.tensorflow.org/install/source_rpi).


## Cross-compile for Raspberry Pi

This has been tested on Ubuntu 16.04.3 64bit and TensorFlow devel docker image
[tensorflow/tensorflow:nightly-devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To cross compile TensorFlow Lite, first install the toolchain and libs:

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
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

You should then be able to compile:

```bash
./tensorflow/lite/tools/make/build_rpi_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a`.


## Compile natively on Raspberry Pi

This has been tested on Raspberry Pi 3b, Raspbian GNU/Linux 9.1 (stretch), gcc version 6.3.0 20170516 (Raspbian 6.3.0-18+rpi1).

Log in to your Raspberry Pi and install the toolchain:

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

You should then be able to compile:

```bash
./tensorflow/lite/tools/make/build_rpi_lib.sh
```

This should compile a static library in:
`tensorflow/lite/tools/make/gen/lib/rpi_armv7/libtensorflow-lite.a`.
