# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is an experimental port of TensorFlow Lite
designed to run machine learning models on microcontrollers and other devices
with only kilobytes of memory.

It doesn't require operating system support, any standard C or C++ libraries, or
dynamic memory allocation. The core runtime fits in 16 KB on an Arm Cortex M3,
and with enough operators to run a speech keyword detection model, takes up a
total of 22 KB.

There are example applications demonstrating the use of microcontrollers for
tasks including wake word detection, gesture classification from accelerometer
data, and image classification using camera data.

## Get started

To try the example applications and learn how to use the API, read
[Get started with microcontrollers](get_started.md).

## Supported platforms

TensorFlow Lite for Microcontrollers is written in C++ 11 and requires a 32-bit
platform. It has been tested extensively with many processors based on the
[Arm Cortex-M Series](https://developer.arm.com/ip-products/processors/cortex-m)
architecture, and has been ported to other architectures including
[ESP32](https://www.espressif.com/en/products/hardware/esp32/overview).

The framework is available as an Arduino library. It can also generate projects
for development environments such as Mbed. It is open source and can be included
in any C++ 11 project.

There are example applications available for the following development boards:

*   [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers)
*   [SparkFun Edge](https://www.sparkfun.com/products/15170)
*   [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)

To learn more about the libraries and examples, see
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

## Developer workflow

To deploy a TensorFlow model to a microcontroller, you will need to follow this
process:

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

    Write your microcontroller code to collect data, perform inference using the
    [C++ library](library.md), and make use of the results.

5.  **Deploy to your device**

    Build and deploy the program to your device.

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
*   Training is not supported

## Next steps

Read [Get started with microcontrollers](get_started.md) to try the example
applications and learn how to use the API.
