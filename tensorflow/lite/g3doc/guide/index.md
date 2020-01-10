# TensorFlow Lite guide

TensorFlow Lite is a set of tools to help developers run TensorFlow models on
mobile, embedded, and IoT devices. It enables on-device machine learning
inference with low latency and a small binary size.

TensorFlow Lite consists of two main components:

-   The [TensorFlow Lite interpreter](inference.md), which runs specially
    optimized models on many different hardware types, including mobile phones,
    embedded Linux devices, and microcontrollers.
-   The [TensorFlow Lite converter](../convert/index.md), which converts
    TensorFlow models into an efficient form for use by the interpreter, and can
    introduce optimizations to improve binary size and performance.

### Machine learning at the edge

TensorFlow Lite is designed to make it easy to perform machine learning on
devices, "at the edge" of the network, instead of sending data back and forth
from a server. For developers, performing machine learning on-device can help
improve:

*   *Latency:* there's no round-trip to a server
*   *Privacy:* no data needs to leave the device
*   *Connectivity:* an Internet connection isn't required
*   *Power consumption:* network connections are power hungry

TensorFlow Lite works with a huge range of devices, from tiny microcontrollers
to powerful mobile phones.

Key Point: The TensorFlow Lite binary is smaller than 1MB when all supported
operators are linked (for 32-bit ARM builds), and less than 300KB when using
only the operators needed for supporting the common image classification models
InceptionV3 and MobileNet.

## Get started

To begin working with TensorFlow Lite on mobile devices, visit
[Get started](get_started.md). If you want to deploy TensorFlow Lite models to
microcontrollers, visit [Microcontrollers](../microcontrollers).

## Key features

*   *[Interpreter](inference.md) tuned for on-device ML*, supporting a set of
    core operators that are optimized for on-device applications, and with a
    small binary size.
*   *Diverse platform support*, covering [Android](android.md) and [iOS](ios.md)
    devices, embedded Linux, and microcontrollers, making use of platform APIs
    for accelerated inference.
*   *APIs for multiple languages* including Java, Swift, Objective-C, C++, and
    Python.
*   *High performance*, with [hardware acceleration](../performance/gpu.md) on
    supported devices, device-optimized kernels, and
    [pre-fused activations and biases](ops_compatibility.md).
*   *Model optimization tools*, including
    [quantization](../performance/post_training_quantization.md), that can
    reduce size and increase performance of models without sacrificing accuracy.
*   *Efficient model format*, using a [FlatBuffer](../convert/index.md) that is
    optimized for small size and portability.
*   *[Pre-trained models](../models)* for common machine learning tasks that can
    be customized to your application.
*   *[Samples and tutorials](https://www.tensorflow.org/examples)* that show you
    how to deploy machine learning models on supported platforms.

## Development workflow

The workflow for using TensorFlow Lite involves the following steps:

1.  **Pick a model**

    Bring your own TensorFlow model, find a model online, or pick a model from
    our [Pre-trained models](../models) to drop in or retrain.

1.  **Convert the model**

    If you're using a custom model, use the
    [TensorFlow Lite converter](../convert/index.md) and a few lines of Python
    to convert it to the TensorFlow Lite format.

1.  **Deploy to your device**

    Run your model on-device with the
    [TensorFlow Lite interpreter](inference.md), with APIs in many languages.

1.  **Optimize your model**

    Use our [Model Optimization Toolkit](../performance/model_optimization.md)
    to reduce your model's size and increase its efficiency with minimal impact
    on accuracy.

To learn more about using TensorFlow Lite in your project, see
[Get started](get_started.md).

## Technical constraints

TensorFlow Lite plans to provide high performance on-device inference for any
TensorFlow model. However, the TensorFlow Lite interpreter currently supports a
limited subset of TensorFlow operators that have been optimized for on-device
use. This means that some models require additional steps to work with
TensorFlow Lite.

To learn which operators are available, see
[Operator compatibility](ops_compatibility.md).

If your model uses operators that are not yet supported by TensorFlow Lite
interpreter, you can use [TensorFlow Select](ops_select.md) to include
TensorFlow operations in your TensorFlow Lite build. However, this will lead to
an increased binary size.

TensorFlow Lite does not currently support on-device training, but it is in our
[Roadmap](roadmap.md), along with other planned improvements.

## Next steps

Want to keep learning about TensorFlow Lite? Here are some next steps:

*   Visit [Get started](get_started.md) to walk through the process of using
    TensorFlow Lite.
*   If you're a mobile developer, visit [Android quickstart](android.md) or
    [iOS quickstart](ios.md).
*   Learn about [TensorFlow Lite for Microcontrollers](../microcontrollers).
*   Explore our [pre-trained models](../models).
*   Try our [example apps](https://www.tensorflow.org/lite/examples).
