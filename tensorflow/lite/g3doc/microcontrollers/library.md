# Understand the C++ library

The TensorFlow Lite for Microcontrollers C++ library is part of the
[TensorFlow repository](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro).
It is designed to be readable, easy to modify, well-tested, easy to integrate,
and compatible with regular TensorFlow Lite.

The following document outlines the basic structure of the C++ library and
provides information about creating your own project.

## File structure

The
[`micro`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro)
root directory has a relatively simple structure. However, since it is located
inside of the extensive TensorFlow repository, we have created scripts and
pre-generated project files that provide the relevant source files in isolation
within various embedded development environments.

### Key files

The most important files for using the TensorFlow Lite for Microcontrollers
interpreter are located in the root of the project, accompanied by tests:

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.h)
    or
    [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h)
    can be used to provide the operations used by the interpreter to run the
    model. Since `all_ops_resolver.h` pulls in every available operation, it
    uses a lot of memory. In production applications, you should use
    `micro_mutable_op_resolver.h` to pull in only the operations your model
    needs.
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_error_reporter.h)
    outputs debug information.
-   [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h)
    contains code to handle and run models.

See [Get started with microcontrollers](get_started_low_level.md) for a
walkthrough of typical usage.

The build system provides for platform-specific implementations of certain
files. These are located in a directory with the platform name, for example
[`sparkfun_edge`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/sparkfun_edge).

Several other directories exist, including:

-   [`kernel`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels),
    which contains operation implementations and the associated code.
-   [`tools`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools),
    which contains build tools and their output.
-   [`examples`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples),
    which contains sample code.

## Start a new project

We recommend using the *Hello World* example as a template for new projects. You
can obtain a version of it for your platform of choice by following the
instructions in this section.

### Use the Arduino library

If you are using Arduino, the *Hello World* example is included in the
`Arduino_TensorFlowLite` Arduino library, which you can download from the
Arduino IDE and in [Arduino Create](https://create.arduino.cc/).

Once the library has been added, go to `File -> Examples`. You should see an
example near the bottom of the list named `TensorFlowLite:hello_world`. Select
it and click `hello_world` to load the example. You can then save a copy of the
example and use it as the basis of your own project.

### Generate projects for other platforms

TensorFlow Lite for Microcontrollers is able to generate standalone projects
that contain all of the necessary source files, using a `Makefile`. The current
supported environments are Keil, Make, and Mbed.

To generate these projects with Make, clone the
[TensorFlow repository](http://github.com/tensorflow/tensorflow) and run the
following command:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

This will take a few minutes, since it has to download some large toolchains for
the dependencies. Once it has finished, you should see some folders created
inside a path like `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/` (the
exact path depends on your host operating system). These folders contain the
generated project and source files.

After running the command, you'll be able to find the *Hello World* projects in
`tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/hello_world`. For
example, `hello_world/keil` will contain the Keil project.

## Run the tests

To build the library and run all of its unit tests, use the following command:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

To run an individual test, use the following command, replacing `<test_name>`
with the name of the test:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

You can find the test names in the project's Makefiles. For example,
`examples/hello_world/Makefile.inc` specifies the test names for the *Hello
World* example.

## Build binaries

To build a runnable binary for a given project (such as an example application),
use the following command, replacing `<project_name>` with the project you wish
to build:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

For example, the following command will build a binary for the *Hello World*
application:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

By default, the project will be compiled for the host operating system. To
specify a different target architecture, use `TARGET=`. The following example
shows how to build the *Hello World* example for the SparkFun Edge:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge hello_world_bin
```

When a target is specified, any available target-specific source files will be
used in place of the original code. For example, the subdirectory
`examples/hello_world/sparkfun_edge` contains SparkFun Edge implementations of
the files `constants.cc` and `output_handler.cc`, which will be used when the
target `sparkfun_edge` is specified.

You can find the project names in the project's Makefiles. For example,
`examples/hello_world/Makefile.inc` specifies the binary names for the *Hello
World* example.

## Optimized kernels

The reference kernels in the root of `tensorflow/lite/micro/kernels` are
implemented in pure C/C++, and do not include platform-specific hardware
optimizations.

Optimized versions of kernels are provided in subdirectories. For example,
`kernels/cmsis-nn` contains several optimized kernels that make use of Arm's
CMSIS-NN library.

To generate projects using optimized kernels, use the following command,
replacing `<subdirectory_name>` with the name of the subdirectory containing the
optimizations:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

You can add your own optimizations by creating a new subfolder for them. We
encourage pull requests for new optimized implementations.

## Generate the Arduino library

A nightly build of the Arduino library is available via the Arduino IDE's
library manager.

If you need to generate a new build of the library, you can run the following
script from the TensorFlow repository:

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

The resulting library can be found in
`tensorflow/lite/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip`.

## Port to new devices

Guidance on porting TensorFlow Lite for Microcontrollers to new platforms and
devices can be found in
[`micro/README.md`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/README.md).
