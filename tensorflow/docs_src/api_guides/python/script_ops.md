# Wraps python functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Script Language Operators

TensorFlow provides allows you to wrap python/numpy functions as
TensorFlow operators.

*   @{tf.py_func}
