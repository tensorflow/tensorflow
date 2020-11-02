# TensorFlow Lite C++ minimal example

This example shows how you can build a simple TensorFlow Lite application.

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

#### Step 3. Create CMake build directory and run CMake tool

```sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```

#### Step 4. Build TensorFlow Lite

In the minimal_build directory,

```sh
cmake --build . -j
```
