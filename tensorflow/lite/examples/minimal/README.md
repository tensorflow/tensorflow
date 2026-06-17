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

#### Step 2. Install libffi7 package(Optional)

It requires libffi7. On Ubuntu 20.10 or later, you can simply run the following
command.

```sh
wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb
sudo dpkg -i libffi7_3.3-4_amd64.deb
```

#### Step 3. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### Step 4. Create CMake build directory and run CMake tool

```sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```

#### Step 5. Build TensorFlow Lite

In the minimal_build directory,

```sh
cmake --build . -j
```

#### Step 5. Run the executable

In the minimal_build directory,

```sh
./minimal <path/to/tflite/model>
```

#### Optional: Link with tensorflowlite_flex library

You may want to link with tensorflowlite_flex library to use TF select Ops in
your model.

First tensorflowlite_flex needs to be compiled using bazel in tensorflow_src
directory: `sh bazel build -c opt --cxxopt='--std=c++17' --config=monolithic
tensorflow/lite/delegates/flex:tensorflowlite_flex`

Then when configuring cmake build
([Step 4](#step-4-create-cmake-build-directory-and-run-cmake-tool)), add the
following option:

```sh
cmake ../tensorflow_src/tensorflow/lite/examples/minimal -DLINK_TFLITE_FLEX="ON"
```

And build: `sh cmake --build . -j`
