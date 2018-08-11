# Tensor Handle Operations

Note: Functions taking `Tensor` arguments can also take anything accepted by
`tf.convert_to_tensor`.

[TOC]

## Tensor Handle Operations

TensorFlow provides several operators that allows the user to keep tensors
"in-place" across run calls.

*   `tf.get_session_handle`
*   `tf.get_session_tensor`
*   `tf.delete_session_tensor`
