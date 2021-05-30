# TensorFlow Lite

TensorFlow Lite is a set of tools that enables on-device machine learning by
helping developers run their models on mobile, embedded, and IoT devices.

### Key features

-   *Optimized for on-device machine learning*, by addressing 5 key constraints:
    latency (there's no round-trip to a server), privacy (no personal data
    leaves the device), connectivity (internet connectivity is not required),
    size (reduced model and binary size) and power consumption (efficient
    inference and a lack of network connections).
-   *Multiple platform support*, covering [Android](android) and [iOS](ios)
    devices, [embedded Linux](python), and
    [microcontrollers](../microcontrollers).
-   *Diverse language support*, which includes Java, Swift, Objective-C, C++,
    and Python.
-   *High performance*, with [hardware acceleration](../performance/delegates)
    and [model optimization](../performance/model_optimization).
-   *End-to-end [examples](../examples)*, for common machine learning tasks such
    as image classification, object detection, pose estimation, question
    answering, text classification, etc. on multiple platforms.

Key Point: The TensorFlow Lite binary is ~1MB when all 125+ supported operators
are linked (for 32-bit ARM builds), and less than 300KB when using only the
operators needed for supporting the common image classification models
InceptionV3 and MobileNet.

## Development workflow

The following guide walks through each step of the workflow and provides links
to further instructions:

Note: Refer to the [performance best practices](../performance/best_practices)
guide for an ideal balance of performance, model size, and accuracy.

### 1. Generate a TensorFlow Lite model

A TensorFlow Lite model is represented in a special efficient portable format
known as [FlatBuffers](https://google.github.io/flatbuffers/){:.external}
(identified by the *.tflite* file extension). This provides several advantages
over TensorFlow's protocol buffer model format such as reduced size (small code
footprint) and faster inference (data is directly accessed without an extra
parsing/unpacking step) that enables TensorFlow Lite to execute efficiently on
devices with limited compute and memory resources.

A TensorFlow Lite model can optionally include *metadata* that has
human-readable model description and machine-readable data for automatic
generation of pre- and post-processing pipelines during on-device inference.
Refer to [Add metadata](../convert/metadata) for more details.

You can generate a TensorFlow Lite model in the following ways:

*   **Use an existing TensorFlow Lite model:** Refer to
    [TensorFlow Lite Examples](../examples) to pick an existing model. *Models
    may or may not contain metadata.*

*   **Create a TensorFlow Lite model:** Use the
    [TensorFlow Lite Model Maker](model_maker) to create a model with your own
    custom dataset. *By default, all models contain metadata.*

*   **Convert a TensorFlow model into a TensorFlow Lite model:** Use the
    [TensorFlow Lite Converter](../convert/index) to convert a TensorFlow model
    into a TensorFlow Lite model. During conversion, you can apply
    [optimizations](../performance/model_optimization) such as
    [quantization](../performance/post_training_quantization) to reduce model
    size and latency with minimal or no loss in accuracy. *By default, all
    models don't contain metadata.*

### 2. Run Inference

*Inference* refers to the process of executing a TensorFlow Lite model on-device
to make predictions based on input data. You can run inference in the following
ways based on the model type:

*   **Models *without* metadata**: Use the
    [TensorFlow Lite Interpreter](inference) API. *Supported on multiple
    platforms and languages such as Java, Swift, C++, Objective-C and Python.*

*   **Models *with* metadata**: You can either leverage the out-of-box APIs
    using the
    [TensorFlow Lite Task Library](../inference_with_metadata/task_library/overview)
    or build custom inference pipelines with the
    [TensorFlow Lite Support Library](../inference_with_metadata/lite_support).
    On android devices, users can automatically generate code wrappers using the
    [Android Studio ML Model Binding](../inference_with_metadata/codegen#mlbinding)
    or the
    [TensorFlow Lite Code Generator](../inference_with_metadata/codegen#codegen).
    *Supported only on Java (Android) while Swift (iOS) and C++ is work in
    progress.*

On Android and iOS devices, you can improve performance using hardware
acceleration. On either platforms you can use a
[GPU Delegate](../performance/gpu), on android you can either use the
[NNAPI Delegate](../performance/nnapi) (for newer devices) or the
[Hexagon Delegate](../performance/hexagon_delegate) (on older devices) and on
iOS you can use the [Core ML Delegate](../performance/coreml_delegate). To add
support for new hardware accelerators, you can
[define your own delegate](../performance/implementing_delegate).

## Get started

You can refer to the following guides based on your target device:

*   **Android and iOS:** Explore the [Android quickstart](android) and
    [iOS quickstart](ios).

*   **Embedded Linux:** Explore the [Python quickstart](python) for embedded
    devices such as [Raspberry Pi](https://www.raspberrypi.org/){:.external} and
    [Coral devices with Edge TPU](https://coral.withgoogle.com/){:.external}, or
    C++ build instructions for [ARM](build_arm).

*   **Microcontrollers:** Explore the
    [TensorFlow Lite for Microcontrollers](../microcontrollers) library for
    microcontrollers and DSPs that contain only a few kilobytes of memory.

## Technical constraints

*   *All TensorFlow models* ***cannot*** *be converted into TensorFlow Lite
    models*, refer to [Operator compatibility](ops_compatibility).

*   *Unsupported on-device training*, however it is on our [Roadmap](roadmap).
