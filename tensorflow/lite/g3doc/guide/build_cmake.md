# Build TensorFlow Lite with CMake

This page describes how to build and use the TensorFlow Lite library with
[CMake](https://cmake.org/) tool.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, macOS Catalina (x86_64), Windows 10 and TensorFlow devel Docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This feature is currently experimental and available since version 2.4
and may change.

### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already
provided in `/tensorflow_src/`.

### Step 3. Create CMake build directory

```sh
mkdir tflite_build
cd tflite_build
```

### Step 4. Run CMake tool with configurations

#### Release build

It generates an optimized release binary by default. If you want to build for
your workstation, simply run the following command.

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### Debug build

If you need to produce a debug build which has symbol information, you need to
provide `-DCMAKE_BUILD_TYPE=Debug` option.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### Cross-compilation for Android

You can use CMake to build Android binaries. You need to install
[Android NDK](https://developer.android.com/ndk) and provide the NDK path with
`-DDCMAKE_TOOLCHAIN_FILE` flag. You also need to set target ABI with
`-DANDROID_ABI` flag.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

#### OpenCL GPU delegate

If your target machine has OpenCL support, you can use
[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) which can
leverage your GPU power.

To configure OpenCL GPU delegate support:

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**Note:** It's experimental and available only on master(r2.5) branch. There
could be compatibility issues. It's only verified with Android devices and
NVidia CUDA OpenCL 1.2.

### Step 5. Build TensorFlow Lite

In the tflite_build directory,

```sh
cmake --build . -j
```

**Note:** This generates a static library `libtensorflow-lite.a` in the current
directory but the library isn't self-contained since all the transitive
dependencies are not included. To use the library properly, you need to create a
CMake project. Please refer the
["Create a CMake project which uses TensorFlow Lite"](#create_a_cmake_project_which_uses_tensorflow_lite)
section.

### Step 6. Build TensorFlow Lite Benchmark Tool and Label Image Example (Optional)

In the tflite_build directory,

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## Available Options to build TensorFlow Lite

Here is the list of available options. You can override it with
`-D<option_name>=[ON|OFF]`. For example, `-DTFLITE_ENABLE_XNNPACK=OFF` to
disable XNNPACK which is enabled by default.

Option Name           | Feature                                  | Default
--------------------- | ---------------------------------------- | ------------
TFLITE_ENABLE_RUY     | Enable RUY matrix multiplication library | OFF
TFLITE_ENABLE_NNAPI   | Enable NNAPI delegate                    | ON (Android)
TFLITE_ENABLE_GPU     | Enable GPU delegate                      | OFF
TFLITE_ENABLE_XNNPACK | Enable XNNPACK delegate                  | ON
TFLITE_ENABLE_MMAP    | Enable MMAP (unsupported on Windows)     | ON

## Create a CMake project which uses TensorFlow Lite

Here is the CMakeLists.txt of
[TFLite minimal example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal).

You need to have add_subdirectory() for TensorFlow Lite directory and link
`tensorflow-lite` with target_link_libraries().

```
cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
```

## Build TensorFlow Lite C library

If you want to build TensorFlow Lite shared library for
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md),
follow [step 1](#step-1-install-cmake-tool) to
[step 3](#step-3-create-cmake-build-directory) first. After that, run the
following commands.

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

This command generates the following shared library in the current directory.

Platform | Library name
-------- | -------------------------
Linux    | libtensorflowlite_c.so
macOS    | libtensorflowlite_c.dylib
Windows  | tensorflowlite_c.dll

**Note:** You need necessary headers (c_api.h, c_api_experimental.h and
common.h) to use the generated shared library.
