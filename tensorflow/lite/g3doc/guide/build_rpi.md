# Build TensorFlow Lite for Raspberry Pi

This page describes how to build the TensorFlow Lite static library for
Raspberry Pi. If you just want to start using TensorFlow Lite to execute your
models, the fastest option is to install the TensorFlow Lite runtime package as
shown in the [Python quickstart](python.md).

**Note:** This page shows how to compile only the C++ static library for
TensorFlow Lite. Alternative install options include: [install just the Python
interpreter API](python.md) (for inferencing only); [install the full
TensorFlow package from pip](https://www.tensorflow.org/install/pip);
or [build the full TensorFlow package](
https://www.tensorflow.org/install/source_rpi).

## Cross-compile for Raspberry Pi

Instruction has been tested on Ubuntu 16.04.3 64-bit PC (AMD64) and TensorFlow devel
docker image
[tensorflow/tensorflow:nightly-devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To cross compile TensorFlow Lite follow the steps:

1. Clone official Raspberry Pi cross-compilation toolchain:

    ```bash
    git clone https://github.com/raspberrypi/tools.git rpi_tools
    ```

2. Clone TensorFlow repository:

    ```bash
    git clone https://github.com/tensorflow/tensorflow.git tensorflow_src

    ```

    **Note:** If you're using the TensorFlow Docker image, the repo is already provided in `/tensorflow_src/`.

3. Run following script at the root of the TensorFlow repository to download all the
build dependencies:

    ```bash
    cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
    ```

    **Note:** You only need to do this once.

4. To build ARMv7 binary for Raspberry Pi 2, 3 and 4 execute:

    ```bash
    PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH ./tensorflow/lite/tools/make/build_rpi_lib.sh
    ```

    **Note:** This should compile a static library in:
    `tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a`.

5. To build ARMv6 binary for Raspberry Pi Zero execute:

    ```bash
    PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH ./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv6
    ```

    **Note:** This should compile a static library in:
    `tensorflow/lite/tools/make/gen/rpi_armv6/lib/libtensorflow-lite.a`.

## Compile natively on Raspberry Pi

Instruction has been tested on Raspberry Pi Zero, Raspbian GNU/Linux 10 (buster), gcc version 8.3.0 (Raspbian 8.3.0-6+rpi1):

To natively compile TensorFlow Lite follow the steps:

1. Log in to your Raspberry Pi and install the toolchain:

    ```bash
    sudo apt-get install build-essential
    ```

2. Clone TensorFlow repository:

    ```bash
    git clone https://github.com/tensorflow/tensorflow.git tensorflow_src

    ```

3. Run following script at the root of the TensorFlow repository to download all the
build dependencies:

    ```bash
    cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
    ```

    **Note:** You only need to do this once.

4. You should then be able to compile TensorFlow Lite with:

    ```bash
    ./tensorflow/lite/tools/make/build_rpi_lib.sh
    ```

    **Note:** This should compile a static library in:
    `tensorflow/lite/tools/make/gen/lib/rpi_armv6/libtensorflow-lite.a`.
