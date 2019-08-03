# Converter command line reference

This page describes how to use the [TensorFlow Lite converter](index.md) using
the command line tool in TensorFlow 2.0. The preferred approach for conversion
is using the [Python API](python_api.md).

[TOC]

## High-level overview

The TensorFlow Lite Converter has a command line tool `tflite_convert` which
supports basic models. Use the `TFLiteConverter` [Python API](python_api.md) for
any conversions involving quantization or any additional parameters (e.g.
signatures in SavedModels or custom objects in Keras models).

## Usage

The following flags specify the input and output files.

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   --saved_model_dir. Type: string. Specifies the full path to the directory
    containing the SavedModel generated in 1.X or 2.0.
*   --keras_model_file. Type: string. Specifies the full path of the HDF5 file
    containing the tf.keras model generated in 1.X or 2.0.

The following is an example usage.

```
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

## Additional instructions

### Building from source

In order to run the latest version of the TensorFlow Lite Converter either
install the nightly build using [pip](https://www.tensorflow.org/install/pip) or
[clone the TensorFlow repository](https://www.tensorflow.org/install/source) and
use `bazel`. An example can be seen below.

```
bazel run //third_party/tensorflow/lite/python:tflite_convert -- \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```
