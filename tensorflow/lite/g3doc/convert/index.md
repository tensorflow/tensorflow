# TensorFlow Lite converter

The TensorFlow Lite converter takes a TensorFlow model and generates a
TensorFlow Lite model file (`.tflite`). The converter supports
[SavedModel directories](https://www.tensorflow.org/guide/saved_model),
[`tf.keras` models](https://www.tensorflow.org/guide/keras/overview), and
[concrete functions](https://tensorflow.org/guide/concrete_function).

Note: This page contains documentation on the converter API for TensorFlow 2.0.
The API for TensorFlow 1.X is available
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/index.md).

## Converting models

In TensorFlow Lite, there are two ways to create a TensorFlow Lite model file:

*   [Python API](python_api.md) (recommended): The Python API makes it easier to
    convert models as part of a model development pipeline and helps mitigate
    [compatibility](../guide/ops_compatibility.md) issues early on.
*   [Command line tool](cmdline.md): The CLI tool supports converting the models
    saved in the supported file formats, the directory containing the SavedModel
    and the HDF5 file containing the
    [`tf.keras` model](https://www.tensorflow.org/guide/keras/overview).

## Device deployment

The TensorFlow Lite model is formatted in
[`FlatBuffer`](https://google.github.io/flatbuffers/). After conversion, The
model file is then deployed to a client device (e.g. mobile, embedded) and run
locally using the TensorFlow Lite interpreter. This conversion process is shown
in the diagram below:

![TFLite converter workflow](../images/convert/workflow.svg)

## MLIR-based conversion

TensorFlow Lite has switched to use a new converter backend, based on MLIR, by
default since TF 2.2 version. The new converter backend provides the following
benefits:

*   Enables conversion of new classes of models, including Mask R-CNN, Mobile
    BERT, and many more
*   Adds support for functional control flow (enabled by default in TensorFlow
    2.x)
*   Tracks original TensorFlow node name and Python code, and exposes them
    during conversion if errors occur
*   Leverages MLIR, Google's cutting edge compiler technology for ML, which
    makes it easier to extend to accommodate feature requests
*   Adds basic support for models with input tensors containing unknown
    dimensions
*   Supports all existing converter functionality

## Getting Help

To get help with issues you may encounter using the TensorFlow Lite converter:

*   Please create a
    [GitHub issue](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)
    with the component label “TFLiteConverter”.
*   If you are using the `allow_custom_ops` feature, please read the
    [Python API](../convert/python_api.md) and
    [Command Line Tool](../convert/cmdline.md) documentation
*   Switch to the old converter by setting `--experimental_new_converter=false`
    (from the [tflite_convert](../convert/cmdline.md) command line tool) or
    `converter.experimental_new_converter=False` (from the
    [Python API](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter))
