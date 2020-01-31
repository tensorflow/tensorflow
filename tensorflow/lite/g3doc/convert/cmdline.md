# Converter command line reference

This page describes how to use the [TensorFlow Lite converter](index.md) using
the command line tool. However, The[Python API](python_api.md) is recommended
for the majority of cases.

Note: This only contains documentation on the command line tool in TensorFlow 2.
Documentation on using the command line tool in TensorFlow 1 is available on
GitHub
([reference](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md),
[example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_examples.md)).

## High-level overview

The TensorFlow Lite Converter has a command line tool named `tflite_convert`,
which supports basic models. Use the [Python API](python_api.md) for any
conversions involving optimizations, or any additional parameters (e.g.
signatures in [SavedModels](https://www.tensorflow.org/guide/saved_model) or
custom objects in
[Keras models](https://www.tensorflow.org/guide/keras/overview)).

## Usage

The following example shows a SavedModel being converted:

```bash
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

The inputs and outputs are specified using the following commonly used flags:

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   `--saved_model_dir`. Type: string. Specifies the full path to the directory
    containing the SavedModel generated in 1.X or 2.X.
*   `--keras_model_file`. Type: string. Specifies the full path of the HDF5 file
    containing the `tf.keras` model generated in 1.X or 2.X.

To use all of the available flags, use the following command:

```bash
tflite_convert --help
```

The following flag can be used for compatibility with the TensorFlow 1.X version
of the converter CLI:

*   `--enable_v1_converter`. Type: bool. Enables user to enable the 1.X command
    line flags instead of the 2.X flags. The 1.X command line flags are
    specified
    [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_reference.md).

## Installing the converter CLI

To obtain the latest version of the TensorFlow Lite converter CLI, we recommend
installing the nightly build using
[pip](https://www.tensorflow.org/install/pip):

```bash
pip install tf-nightly
```

Alternatively, you can
[clone the TensorFlow repository](https://www.tensorflow.org/install/source) and
use `bazel` to run the command:

```
bazel run //tensorflow/lite/python:tflite_convert -- \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```
