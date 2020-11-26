# Build TensorFlow Lite with CMake

This page describes how to build the TensorFlow Lite static library with CMake
tool.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, TensorFlow devel docker image and Windows 10.
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This is an experimental that is subject to change.

**Note:** The following are not currently supported: iOS, Tests and
Host Tools (i.e analysis tools etc.)

#### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

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

It generates release binary by default. If you need to produce debug builds, you
need to provide '-DCMAKE_BUILD_TYPE=Debug' option.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

If you want to configure Android build with GPU delegate support,

```sh
mkdir tflite_build
cd tflite_build
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DTFLITE_ENABLE_GPU=ON ../tensorflow_src/tensorflow/lite
```


#### Step 4. Build TensorFlow Lite

In the tflite_build directory,

```sh
cmake --build . -j
```

Or

```sh
make -j
```


**Note:** This should compile a static library `libtensorflow-lite.a` in the
current directory.


#### Step 5. Build TensorFlow Lite Benchmark Tool

In the tflite_build directory,

```sh
cmake --build . -j -t benchmark_model
```

Or

```sh
make benchmark_model -j
```
