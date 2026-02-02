# TensorFlow Graph Editor

The TensorFlow Graph Editor libray which allows for modification of an existing
tf.Graph instance.

## Overview of the modules

* util.py: utility functions
* select.py: selection functions, allowing for various selection method of
  tensors and operations.
* subgraph.py: the SubGraphView class, allowing to manipulate subgraph of a
  TensorFlow tf.Graph instance.
* transform.py: the Transformer class, allowing to tranform a subgraph into
  another one.
* edit.py: various editing function operating on subgraph.
