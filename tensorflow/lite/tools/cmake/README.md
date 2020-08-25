# Build TensorFlow Lite with CMake

This page describes how to build the TensorFlow Lite static library with CMake
tool.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
and TensorFlow devel docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This is an experimental that is subject to change.

**Note:** The following are not currently supported: Android, iOS, Tests and
Host Tools (i.e benchmark / analysis tools etc.)

#### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubunutu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow [the offcial cmake installation guide](https://cmake.org/install/)

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already
provided in `/tensorflow_src/`.

#### Step 3. Create CMake build directory and run CMake tool

```sh
mkdir tflite_build
cd tflite_build
cmake ../tensorflow_src/tensorflow/lite
```

#### Step 4. Build TensorFlow Lite

```sh
cmake --build . -j
```

**Note:** This should compile a static library `libtensorflow-lite.a` in the
current directory.
