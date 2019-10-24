# Converter command line reference

This page describes how to use the [TensorFlow Lite converter](index.md) using
the command line tool. The preferred approach for conversion is using the
[Python API](python_api.md).

Note: This only contains documentation on the command line tool in TensorFlow 2.
Documentation on using the command line tool in TensorFlow 1 is available on
GitHub
([reference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md),
[example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_examples.md)).

[TOC]

## High-level overview

The TensorFlow Lite Converter has a command line tool `tflite_convert` which
supports basic models. Use the `TFLiteConverter` [Python API](python_api.md) for
any conversions involving quantization or any additional parameters (e.g.
signatures in [SavedModels](https://www.tensorflow.org/guide/saved_model) or
custom objects in
[Keras models](https://www.tensorflow.org/guide/keras/overview)).

## Usage

The following flags specify the input and output files.

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   `--saved_model_dir`. Type: string. Specifies the full path to the directory
    containing the SavedModel generated in 1.X or 2.X.
*   `--keras_model_file`. Type: string. Specifies the full path of the HDF5 file
    containing the `tf.keras` model generated in 1.X or 2.X.

The following is an example usage.

```
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

In addition to the input and output flags, the converter contains the following
flag.

*   `--enable_v1_converter`. Type: bool. Enables user to enable the 1.X command
    line flags instead of the 2.X flags. The 1.X command line flags are
    specified
    [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md).

## Additional instructions

### Building from source

In order to run the latest version of the TensorFlow Lite Converter either
install the nightly build using [pip](https://www.tensorflow.org/install/pip) or
[clone the TensorFlow repository](https://www.tensorflow.org/install/source) and
use `bazel`. An example can be seen below.

```
bazel run //tensorflow/lite/python:tflite_convert -- \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```
