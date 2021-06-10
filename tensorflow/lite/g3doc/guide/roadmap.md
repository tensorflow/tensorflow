# TensorFlow Lite Roadmap

**Updated: May, 2021**

The following represents a high level overview of our roadmap. You should be
aware that this roadmap may change at any time and the order below does not
reflect any type of priority.

We break our roadmap into four key segments: usability, performance,
optimization and portability. We strongly encourage you to comment on our
roadmap and provide us feedback in the
[TensorFlow Lite discussion group](https://groups.google.com/a/tensorflow.org/g/tflite).

## Usability

*   **Expanded ops coverage**
    *   Add targeted ops based on user feedback.
    *   Add targeted op sets for specific domains and areas including Random
        ops, base Keras layer ops, hash tables, select training ops.
*   **More assistive tooling**
    *   Provide TensorFlow graph annotations and compatibility tools to validate
        TFLite and hardware accelerator compatibility during-training and
        after-conversion.
    *   Allow targeting and optimizing for specific accelerators during
        conversion.
*   **On-device training**
    *   Support on-device training for personalization and transfer learning,
        including a Colab demonstrating end-to-end usage.
    *   Support variable/resource types (both for inference and training)
    *   Support converting and executing graphs with multiple function (or
        signature) entry-points.
*   **Enhanced Android Studio integration**
    *   Drag and drop TFLite models into Android Studio to generate model
        interfaces.
    *   Improve Android Studio profiling support, including memory profiling.
*   **Model Maker**
    *   Support newer tasks, including object detection, recommendation, and
        audio classification, covering a wide collection of common usage.
    *   Support more data sets to make transfer learning easier.
*   **Task Library**
    *   Support more model types (e.g. audio, NLP) with associated pre and post
        processing capabilities.
    *   Update more reference examples with Task APIs.
    *   Support out-of-the-box acceleration for all tasks.
*   **More SOTA models and examples**
    *   Add more examples (e.g. audio, NLP, structure-data related) to
        demonstrate model usage as well as new features and APIs, covering
        different platforms.
    *   Create shareable backbone models for on-device to reduce training and
        deployment costs.
*   **Seamless deployment across multiple platforms**
    *   Run TensorFlow Lite models on the web.
*   **Improved cross-platform support**
    *   Extend and improve APIs for Java on Android, Swift on iOS, Python on
        RPi.
    *   Enhance CMake support (e.g., broader accelerator support).
*   **Better frontend support**
    *   Improve compatibility with various authoring frontends, including Keras,
        tf.numpy.

## Performance

*   **Better tooling**
    *   Public dashboard for tracking performance gains with each release.
    *   Tooling for better understanding graph compatibility with target
        accelerators.
*   **Improved CPU performance**
    *   XNNPack enabled by default for faster floating point inference.
    *   End-to-end half precision (float16) support with optimized kernels.
*   **Updated NN API support**
    *   Full support for newer Android version NN API features, ops, and types.
*   **GPU optimizations**
    *   Improved startup time with delegate serialization support.
    *   Hardware buffer interop for zero-copy inference.
    *   Wider availability of on device acceleration.
    *   Better op coverage.

## Optimization

*   **Quantization**

    *   Selective post-training quantization to exclude certain layers from
        quantization.
    *   Quantization debugger to inspect quantization error losses per each
        layer.
    *   Applying quantization-aware training on more model coverage e.g.
        TensorFlow Model Garden.
    *   Quality and performance improvements for post-training dynamic-range
        quantization.
    *   Tensor Compression API to allow compression algorithms such as SVD.

*   **Pruning / sparsity**

    *   Combine configurable training-time (pruning + quantization-aware
        training) APIs.
    *   Increase sparity application on TF Model Garden models.
    *   Sparse model execution support in TensorFlow Lite.

## Portability

*   **Microcontroller Support**
    *   Add support for a range of 32-bit MCU architecture use cases for speech
        and image classification.
    *   Audio Frontend: In-graph audio pre-processing and acceleration support
    *   Sample code and models for vision and audio data.
