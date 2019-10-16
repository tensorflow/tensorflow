# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is an experimental port of TensorFlow Lite
aimed at microcontrollers and other devices with only kilobytes of memory.

It is designed to be portable even to "bare metal" systems, so it doesn't
require operating system support, any standard C or C++ libraries, or dynamic
memory allocation. The core runtime fits in 16KB on a Cortex M3, and with enough
operators to run a speech keyword detection model, takes up a total of 22KB.

## Get started

To quickly get up and running with TensorFlow Lite for Microcontrollers, read
[Get started with microcontrollers](get_started.md).

## Why microcontrollers are important

Microcontrollers are typically small, low-powered computing devices that are
often embedded within hardware that requires basic computation, including
household appliances and Internet of Things devices. Billions of
microcontrollers are manufactured each year.

Microcontrollers are often optimized for low energy consumption and small size,
at the cost of reduced processing power, memory, and storage. Some
microcontrollers have features designed to optimize performance on machine
learning tasks.

By running machine learning inference on microcontrollers, developers can add AI
to a vast range of hardware devices without relying on network connectivity,
which is often subject to bandwidth and power constraints and results in high
latency. Running inference on-device can also help preserve privacy, since no
data has to leave the device.

## Features and components

*   C++ API, with runtime that fits in 16KB on a Cortex M3
*   Uses standard TensorFlow Lite
    [FlatBuffer](https://google.github.io/flatbuffers/) schema
*   Pre-generated project files for popular embedded development platforms, such
    as Arduino, Keil, and Mbed
*   Optimizations for several embedded platforms
*   [Sample code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech)
    demonstrating spoken hotword detection

## Developer workflow

This is the process for deploying a TensorFlow model to a microcontroller:

1.  **Create or obtain a TensorFlow model**

    The model must be small enough to fit on your target device after
    conversion, and it can only use
    [supported operations](build_convert.md#operation-support). If you want to
    use operations that are not currently supported, you can provide your own
    implementations.

2.  **Convert the model to a TensorFlow Lite FlatBuffer**

    You will convert your model into the standard TensorFlow Lite format using
    the [TensorFlow Lite converter](build_convert.md#model-conversion). You may
    wish to output a quantized model, since these are smaller in size and more
    efficient to execute.

3.  **Convert the FlatBuffer to a C byte array**

    Models are kept in read-only program memory and provided in the form of a
    simple C file. Standard tools can be used to
    [convert the FlatBuffer into a C array](build_convert.md#convert-to-a-c-array).

4.  **Integrate the TensorFlow Lite for Microcontrollers C++ library**

    Write your microcontroller code to perform inference using the
    [C++ library](library.md).

5.  **Deploy to your device**

    Build and deploy the program to your device.

## Supported platforms

One of the challenges of embedded software development is that there are a lot
of different architectures, devices, operating systems, and build systems. We
aim to support as many of the popular combinations as we can, and make it as
easy as possible to add support for others.

If you're a product developer, we have build instructions or pre-generated
project files that you can download for the following platforms:

Device                                                                                         | Mbed                                                                           | Keil                                                                           | Make/GCC
---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------
[STM32F746G Discovery Board](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)     | [Download](https://drive.google.com/open?id=1OtgVkytQBrEYIpJPsE8F6GUKHPBS3Xeb) | -                                                                              | [Download](https://drive.google.com/open?id=1u46mTtAMZ7Y1aD-He1u3R8AE4ZyEpnOl)
["Blue Pill" STM32F103-compatible development board](https://github.com/google/stm32_bare_lib) | -                                                                              | -                                                                              | [Instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#building-for-the-blue-pill-stm32f103-using-make)
[Ambiq Micro Apollo3Blue EVB using Make](https://ambiqmicro.com/apollo-ultra-low-power-mcus/)  | -                                                                              | -                                                                              | [Instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#building-for-ambiq-micro-apollo3blue-evb-using-make)
[Generic Keil uVision Projects](http://www2.keil.com/mdk5/uvision/)                            | -                                                                              | [Download](https://drive.google.com/open?id=1Lw9rsdquNKObozClLPoE5CTJLuhfh5mV) | -
[Eta Compute ECM3531 EVB](https://etacompute.com/)                                             | -                                                                              | -                                                                              | [Instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#Building-for-the-Eta-Compute-ECM3531-EVB-using-Make)

If your device is not yet supported, it may not be difficult add support. You
can learn about that process in
[README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#how-to-port-tensorflow-lite-micro-to-a-new-platform).

### Portable reference code

If you don't have a particular microcontroller platform in mind yet, or just
want to try out the code before beginning porting, the easiest way to begin is
by
[downloading the platform-agnostic reference code](https://drive.google.com/open?id=1cawEQAkqquK_SO4crReDYqf_v7yAwOY8).

There is a series of folders inside the archive, with each one containing just
the source files you need to build one binary. There is a simple Makefile for
each folder, but you should be able to load the files into almost any IDE and
build them. There is also a [Visual Studio Code](https://code.visualstudio.com/)
project file already set up, so you can easily explore the code in a
cross-platform IDE.

## Goals

Our design goals are to make the framework readable, easy to modify,
well-tested, easy to integrate, and fully compatible with TensorFlow Lite via a
consistent file schema, interpreter, API, and kernel interface.

You can read more about the design in
[goals and tradeoffs](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#goals).

## Limitations

TensorFlow Lite for Microcontrollers is designed for the specific constraints of
microcontroller development. If you are working on more powerful devices (for
example, an embedded Linux device like the Raspberry Pi), the standard
TensorFlow Lite framework might be easier to integrate.

The following limitations should be considered:

*   Support for a [limited subset](build_convert.md#operation-support) of
    TensorFlow operations
*   Support for a limited set of devices
*   Low-level C++ API requiring manual memory management
