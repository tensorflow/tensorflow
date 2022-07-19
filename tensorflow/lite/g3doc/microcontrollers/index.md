# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is designed to run machine learning models
on microcontrollers and other devices with only few kilobytes of memory. The
core runtime just fits in 16 KB on an Arm Cortex M3 and can run many basic
models. It doesn't require operating system support, any standard C or C++
libraries, or dynamic memory allocation.

Note: The
[TensorFlow Lite for Microcontrollers Experiments](https://experiments.withgoogle.com/collection/tfliteformicrocontrollers)
features work by developers combining Arduino and TensorFlow to create awesome
experiences and tools. Check out the site for inspiration to create your own
TinyML projects.

## Why microcontrollers are important

Microcontrollers are typically small, low-powered computing devices that are
embedded within hardware that requires basic computation. By bringing machine
learning to tiny microcontrollers, we can boost the intelligence of billions of
devices that we use in our lives, including household appliances and Internet of
Things devices, without relying on expensive hardware or reliable internet
connections, which is often subject to bandwidth and power constraints and
results in high latency. This can also help preserve privacy, since no data
leaves the device. Imagine smart appliances that can adapt to your daily
routine, intelligent industrial sensors that understand the difference between
problems and normal operation, and magical toys that can help kids learn in fun
and delightful ways.

## Supported platforms

TensorFlow Lite for Microcontrollers is written in C++ 11 and requires a 32-bit
platform. It has been tested extensively with many processors based on the
[Arm Cortex-M Series](https://developer.arm.com/ip-products/processors/cortex-m)
architecture, and has been ported to other architectures including
[ESP32](https://www.espressif.com/en/products/hardware/esp32/overview). The
framework is available as an Arduino library. It can also generate projects for
development environments such as Mbed. It is open source and can be included in
any C++ 11 project.

The following development boards are supported:

*   [Arduino Nano 33 BLE Sense](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense-with-headers)
*   [SparkFun Edge](https://www.sparkfun.com/products/15170)
*   [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
*   [Adafruit EdgeBadge](https://www.adafruit.com/product/4400)
*   [Adafruit TensorFlow Lite for Microcontrollers Kit](https://www.adafruit.com/product/4317)
*   [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all)
*   [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview)
*   [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview)
*   [Wio Terminal: ATSAMD51](https://www.seeedstudio.com/Wio-Terminal-p-4509.html)
*   [Himax WE-I Plus EVB Endpoint AI Development Board](https://www.sparkfun.com/products/17256)
*   [Synopsys DesignWare ARC EM Software Development Platform](https://www.synopsys.com/dw/ipdir.php?ds=arc-em-software-development-platform)
*   [Sony Spresense](https://developer.sony.com/develop/spresense/)

## Explore the examples

Each example application is on
[Github](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples)
and has a `README.md` file that explains how it can be deployed to its supported
platforms. Some examples also have end-to-end tutorials using a specific
platform, as given below:

*   [Hello World](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world) -
    Demonstrates the absolute basics of using TensorFlow Lite for
    Microcontrollers
    *   [Tutorial using any supported device](get_started_low_level.md)
*   [Micro speech](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech) -
    Captures audio with a microphone to detect the words "yes" and "no"
    *   [Tutorial using SparkFun Edge](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow/#0)
*   [Magic wand](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/magic_wand) -
    Captures accelerometer data to classify three different physical gestures
    *   [Tutorial using Arduino Nano 33 BLE Sense](https://codelabs.developers.google.com/codelabs/ai-magicwand/#0)
*   [Person detection](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/person_detection) -
    Captures camera data with an image sensor to detect the presence or absence
    of a person

## Workflow

The following steps are required to deploy and run a TensorFlow model on a
microcontroller:

1.  **Train a model**:
    *   *Generate a small TensorFlow model* that can fit your target device and
        contains [supported operations](build_convert.md#operation-support).
    *   *Convert to a TensorFlow Lite model* using the
        [TensorFlow Lite converter](build_convert.md#model-conversion).
    *   *Convert to a C byte array* using
        [standard tools](build_convert.md#convert-to-a-c-array) to store it in a
        read-only program memory on device.
2.  **Run inference** on device using the [C++ library](library.md) and process
    the results.

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
*   On device training is not supported

## Next steps

*   [Get started with microcontrollers](get_started_low_level.md) to try the
    example application and learn how to use the API.
*   [Understand the C++ library](library.md) to learn how to use the library in
    your own project.
*   [Build and convert models](build_convert.md) to learn more about training
    and converting models for deployment on microcontrollers.
