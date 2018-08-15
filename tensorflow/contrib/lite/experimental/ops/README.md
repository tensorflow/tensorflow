The TensorFlow Lite ops API provides functions to author TFLite ops in a
TensorFlow directily.

Each python function creates a TensorFlow Function node in the graph.
The node has a `_tflite_function_name` attribute to annotate which TensorFlow
Lite op should be used.

The graph can be used in TensorFlow for training and inference directly.
After training is done, user can freeze the graph, and use the converter under
the `experimental/pb2lite` directory to convert the graph to TensorFlow Lite
model format.

Warning: Everything in this directory is experimental and highly subject to
changes.
