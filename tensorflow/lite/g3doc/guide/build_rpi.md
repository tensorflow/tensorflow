# Build TensorFlow Lite for Raspberry Pi

This page describes how to build the TensorFlow Lite static and shared libraries
for Raspberry Pi. If you just want to start using TensorFlow Lite to execute
your models, the fastest option is to install the TensorFlow Lite runtime
package as shown in the [Python quickstart](python.md).

**Note:** This page shows how to compile the C++ static and shared libraries for
TensorFlow Lite. Alternative install options include:
[install just the Python interpreter API](python.md) (for inferencing only);
[install the full TensorFlow package from pip](https://www.tensorflow.org/install/pip);
or
[build the full TensorFlow package](https://www.tensorflow.org/install/source_rpi).

**Note:** This page only covers 32-bit builds. If you're looking for 64-bit
builds, check [Build for ARM64](build_arm64.md) page.

**Note:** Cross-compile ARM with CMake is available. Please check
[this](https://www.tensorflow.org/lite/guide/build_cmake_arm).

## Cross-compile for Raspberry Pi with Make

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
and TensorFlow devel docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

To cross compile TensorFlow Lite follow the steps:

#### Step 1. Clone official Raspberry Pi cross-compilation toolchain

```sh
git clone https://github.com/raspberrypi/tools.git rpi_tools
```

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already
provided in `/tensorflow_src/`.

#### Step 3. Run following script at the root of the TensorFlow repository to download

all the build dependencies:

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**Note:** You only need to do this once.

#### Step 4a. To build ARMv7 binary for Raspberry Pi 2, 3 and 4

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh
```

**Note:** This should compile a static library in:
`tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a`.

You can add additional Make options or target names to the `build_rpi_lib.sh`
script since it's a wrapper of Make with TFLite
[Makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/make/Makefile).
Here are some possible options:

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh clean # clean object files
./tensorflow/lite/tools/make/build_rpi_lib.sh -j 16 # run with 16 jobs to leverage more CPU cores
./tensorflow/lite/tools/make/build_rpi_lib.sh label_image # # build label_image binary
```

#### Step 4b. To build ARMv6 binary for Raspberry Pi Zero

```sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH \
  ./tensorflow/lite/tools/make/build_rpi_lib.sh TARGET_ARCH=armv6
```

**Note:** This should compile a static library in:
`tensorflow/lite/tools/make/gen/rpi_armv6/lib/libtensorflow-lite.a`.

## Compile natively on Raspberry Pi

The following instructions have been tested on Raspberry Pi Zero, Raspberry Pi
OS GNU/Linux 10 (Buster), gcc version 8.3.0 (Raspbian 8.3.0-6+rpi1):

To natively compile TensorFlow Lite follow the steps:

#### Step 1. Log in to your Raspberry Pi and install the toolchain

```sh
sudo apt-get install build-essential
```

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### Step 3. Run following script at the root of the TensorFlow repository to download all the build dependencies

```sh
cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh
```

**Note:** You only need to do this once.

#### Step 4. You should then be able to compile TensorFlow Lite with:

```sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
```

**Note:** This should compile a static library in:
`tensorflow/lite/tools/make/gen/lib/rpi_armv6/libtensorflow-lite.a`.

## Cross-compile for armhf with Bazel

You can use
[ARM GCC toolchains](https://github.com/tensorflow/tensorflow/tree/master/third_party/toolchains/embedded/arm-linux)
with Bazel to build an armhf shared library which is compatible with Raspberry
Pi 2, 3 and 4.

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

#### Step 3. Build ARMv7 binary for Raspberry Pi 2, 3 and 4

##### C library

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite/c:libtensorflowlite_c.so
```

Check
[TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c)
page for the detail.

##### C++ library

```bash
bazel build --config=elinux_armhf -c opt //tensorflow/lite:libtensorflowlite.so
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
