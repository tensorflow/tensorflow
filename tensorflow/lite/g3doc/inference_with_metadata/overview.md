# TensorFlow Lite inference with metadata

Inferencing [models with metadata](../convert/metadata.md) can be as easy as
just a few lines of code. TensorFlow Lite metadata contains a rich description
of what the model does and how to use the model. It can empower code generators
to automatically generate the inference code for you, such as using the
[Android Studio ML Binding feature](codegen.md#mlbinding) or
[TensorFlow Lite Android code generator](codegen.md#codegen). It can also be
used to configure your custom inference pipeline.

## Tools and libraries

TensorFlow Lite provides varieties of tools and libraries to serve different
tiers of deployment requirements as follows:

### Generate model interface with Android code generators

There are two ways to automatically generate the necessary Android wrapper code
for TensorFlow Lite model with metadata:

1.  [Android Studio ML Model Binding](codegen.md#mlbinding) is tooling available
    within Android Studio to import TensorFlow Lite model through a graphical
    interface. Android Studio will automatically configure settings for the
    project and generate wrapper classes based on the model metadata.

2.  [TensorFlow Lite Code Generator](codegen.md#codegen) is an executable that
    generates model interface automatically based on the metadata. It currently
    supports Android with Java. The wrapper code removes the need to interact
    directly with `ByteBuffer`. Instead, developers can interact with the
    TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`.
    Android Studio users can also get access to the codegen feature through
    [Android Studio ML Binding](codegen.md#mlbinding).

### Leverage out-of-box APIs with the TensorFlow Lite Task Library

[TensorFlow Lite Task Library](task_library/overview.md) provides optimized
ready-to-use model interfaces for popular machine learning tasks, such as image
classification, question and answer, etc. The model interfaces are specifically
designed for each task to achieve the best performance and usability. Task
Library works cross-platform and is supported on Java, C++, and Swift.

### Build custom inference pipelines with the TensorFlow Lite Support Library

[TensorFlow Lite Support Library](lite_support.md) is a cross-platform library
that helps to customize model interface and build inference pipelines. It
contains varieties of util methods and data structures to perform pre/post
processing and data conversion. It is also designed to match the behavior of
TensorFlow modules, such as TF.Image and TF.Text, ensuring consistency from
training to inferencing.

## Explore pretrained models with metadata

Browse
[TensorFlow Lite hosted models](https://www.tensorflow.org/lite/guide/hosted_models)
and [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) to download
pretrained models with metadata for both vision and text tasks. Also see
different options of
[visualizing the metadata](../convert/metadata.md#visualize-the-metadata).

## TensorFlow Lite Support GitHub repo

Visit the
[TensorFlow Lite Support GitHub repo](https://github.com/tensorflow/tflite-support)
for more examples and source code. Let us know your feedback by creating a
[new GitHub issue](https://github.com/tensorflow/tflite-support/issues/new).
