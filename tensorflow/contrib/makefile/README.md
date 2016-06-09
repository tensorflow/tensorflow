### TensorFlow Makefile

The recommended way to build TensorFlow from source is using the Bazel
open-source build system. Sometimes this isn't possible though:

 - The system may not have the RAM or processing power to support Bazel.
 - Bazel dependencies might not be available.
 - You may want to cross-compile for an unsupported target system.

This experimental project supplies a Makefile automatically derived from the
dependencies listed in the Bazel project, that can be used with GNU's make tool.
It offers the ability to compile the core C++ runtime into a static library, but
doesn't include more advanced features like Python or other language bindings,
or GPU support.

## Building

To compile the library and an example program using it, first pull the
dependencies:

```bash
tensorflow/contrib/makefile/download_dependencies.sh
```

You should only need to do this step once, it puts required libraries like Eigen
in the `tensorflow/contrib/makefile/downloads/` folder. You will also need to
make sure you have a version of [protobuf 3](https://github.com/google/protobuf)
installed on your system, either through package management or building from
source.

Then you can build the project:

```bash
make -f tensorflow/contrib/makefile/Makefile
```

This should compile a static library in 
`tensorflow/contrib/makefile/gen/lib/tf_lib.a`, and create an example executable
at `tensorflow/contrib/makefile/gen/bin/benchmark`. To run the executable, use:

```bash
tensorflow/contrib/makefile/gen/bin/benchmark --graph=tensorflow_inception_graph.pb
```

You should download the example graph from [https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip).

## Supported Systems

The script has been tested on Ubuntu, OS X, Android, and iOS. If you look in the
Makefile itself, you'll see it's broken up into host and target sections. If you
are cross-compiling, you should look at customizing the target settings to match
what you need for the system you're aiming at.

## Android

For Android, you'll need to explicitly specify that as the target, and supply
the location of the NDK toolchain on the command line, for example:

```bash
make -f tensorflow/contrib/makefile/Makefile \
TARGET=ANDROID \
ANDROID_NDK_DIR=$(HOME)/toolchains/clang-21-stl-gnu
```

You'll also need a compiled version of the protobuf libraries for Android. You
can use the helper script at `compile_android_protobuf.sh` to create these.

## iOS

For iOS you'll need to first run `compile_ios_protobuf.sh` to build iOS versions
of the protobuf libraries. Once that's complete, you can run the makefile
specifying iOS as the target, along with the architecture you want to build for:

```bash
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS \
IOS_ARCH=ARM64
```

This will build the library for a single architecture, and the benchmark
program. Since the benchmark is command-line only, you'll need to load the
static library into an Xcode app project to use it.

To build a complete universal library for iOS, containing all architectures,
you will need to run `compile_ios_protobuf.sh` followed by
`compile_ios_tensorflow.sh`. This creates a library in 
`tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a` that you can link any
xcode project against. Here are complete build instructions:

Grab the source code for TensorFlow:

```bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
```

Download dependencies like Eigen and Protobuf:

```bash
tensorflow/contrib/makefile/download_dependencies.sh
```

Build and install the host (OS X) copy of protobuf.

```bash
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure
make
sudo make install
cd ../../../../..
```

Build the iOS native versions of protobuf:

```bash
tensorflow/contrib/makefile/compile_ios_protobuf.sh
```

Build all iOS architectures for TensorFlow:

```bash
tensorflow/contrib/makefile/compile_ios_tensorflow.sh
```

You will need to use -force_load in the linker flags
section of the build settings to pull in the global constructors that are used
to register ops and kernels. 

The example Xcode project in tensorflow/contrib/ios_example shows how to use the
static library in a simple app.

## Raspberry Pi

Building on the Raspberry Pi is similar to a normal Linux system, though we
recommend starting by compiling and installing protobuf:

```bash
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh 
./configure
make
sudo make install
cd ../../../../..
```

Once that's done, you can use make to build the library and example:

```bash
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI OPTFLAGS="-Os"
```

If you're only interested in building for Raspberry Pi's 2 and 3, you can supply
some extra optimization flags to give you code that will run faster:

```bash
make -f tensorflow/contrib/makefile/Makefile HOST_OS=PI TARGET=PI \
OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize"
```

## Dependencies

The Makefile loads in a list of dependencies stored in text files. These files
are generated from the main Bazel build by running 
`tensorflow/contrib/makefile/gen_file_lists.sh`. You'll need to re-run this i
you make changes to the files that are included in the build.

Header dependencies are not automatically tracked by the Makefile, so if you
make header changes you will need to run this command to recompile cleanly:

```bash
make -f tensorflow/contrib/makefile/Makefile clean
```
