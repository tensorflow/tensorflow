# Understand the C++ library

The TensorFlow Lite for Microcontrollers C++ library is part of the
[TensorFlow repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro).
It is designed to be readable, easy to modify, well-tested, easy to integrate,
and compatible with regular TensorFlow Lite.

The following document will outline the basic structure of the C++ library,
provide the commands required for compilation, and give an overview of how to
port to new devices.

The
[README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#how-to-port-tensorflow-lite-micro-to-a-new-platform)
contains more in-depth information on all of these topics.

## File structure

The
[`micro`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
root directory has a relatively simple structure. However, since it is located
inside of the extensive TensorFlow repository, we have created scripts and
pre-generated project files that provide the relevant source files in isolation
within various embedded development environments such as Arduino, Keil, Make,
and Mbed.

### Key files

The most important files for using the TensorFlow Lite for Microcontrollers
interpreter are located in the root of the project, accompanied by tests:

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h)
    provides the operations used by the interpreter to run the model.
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_error_reporter.h)
    outputs debug information.
-   [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_interpreter.h)
    contains code to handle and run models.

See [Get started with microcontrollers](get_started.md) for a walkthrough of
typical usage.

The build system provides for platform-specific implementations of certain
files. These are located in a directory with the platform name, for example
[`sparkfun_edge`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/sparkfun_edge).

Several other directories exist, including:

-   [`kernel`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/kernels),
    which contains operation implementations and the associated code.
-   [`tools`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/tools),
    which contains build tools and their output.
-   [`examples`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples),
    which contains sample code.

### Generate project files

The project's `Makefile` is able to generate standalone projects containing all
necessary source files that can be imported into embedded development
environments. The current supported environments are Arduino, Keil, Make, and
Mbed.

Note: We host prebuilt projects for some of these environments. See
[Supported platforms](overview.md#supported-platforms) to download.

To generate these projects with Make, use the following command:

```bash
make -f tensorflow/lite/experimental/micro/tools/make/Makefile generate_projects
```

This will take a few minutes, since it has to download some large toolchains for
the dependencies. Once it has finished, you should see some folders created
inside a path like
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/` (the exact
path depends on your host operating system). These folders contain the generated
project and source files. For example,
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/keil`
contains the Keil uVision targets.

## Build the library

If you are using a generated project, see its included README for build
instructions.

To build the library and run tests from the main TensorFlow repository, run the
following commands:

1.  Clone the TensorFlow repository from GitHub to a convenient place.

    ```bash
    git clone --depth 1 https://github.com/tensorflow/tensorflow.git
    ```

1.  Enter the directory that was created in the previous step.

    ```bash
    cd tensorflow
    ```

1.  Invoke the `Makefile` to build the project and run tests. Note that this
    will download all required dependencies:

    ```bash
    make -f tensorflow/lite/experimental/micro/tools/make/Makefile test
    ```

## Port to new devices

Guidance on porting TensorFlow Lite for Microcontrollers to new platforms and
devices can be found in
[README.md](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#how-to-port-tensorflow-lite-micro-to-a-new-platform).
