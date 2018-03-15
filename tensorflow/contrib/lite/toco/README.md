# The TensorFlow Lite Optimizing Converter

The TensorFlow Lite Optimizing Converter's most typical use is converting from the TensorFlow GraphDef to the TensorFlow Lite
format, but it supports much more than that.

## Usage documentation

Usage information is given in these documents:

*   [Command-line examples](g3doc/cmdline_examples.md)
*   [Command-line reference](g3doc/cmdline_reference.md)
*   [Python API](g3doc/python_api.md)

## Design documentation

Coming soon!

## Where the converter fits in the TensorFlow landscape

In the typical case, an application developer is using TensorFlow to design and
train models, then uses TensorFlow's freeze_graph.py to generate a frozen
inference graph, then uses the converter to convert that into a TensorFlow Lite flatbuffer file,
then ships that file to client devices where the TensorFlow Lite interpreter handles them
on-device. This is represented in the following diagram:

![drawing](https://storage.googleapis.com/download.tensorflow.org/example_images/tensorflow_landscape.svg)
