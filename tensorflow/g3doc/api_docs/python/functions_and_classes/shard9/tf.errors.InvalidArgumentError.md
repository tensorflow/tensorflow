Raised when an operation receives an invalid argument.

This may occur, for example, if an operation is receives an input
tensor that has an invalid value or shape. For example, the
[`tf.matmul()`](../../api_docs/python/math_ops.md#matmul) op will raise this
error if it receives an input that is not a matrix, and the
[`tf.reshape()`](../../api_docs/python/array_ops.md#reshape) op will raise
this error if the new shape does not match the number of elements in the input
tensor.

- - -

#### `tf.errors.InvalidArgumentError.__init__(node_def, op, message)` {#InvalidArgumentError.__init__}

Creates an `InvalidArgumentError`.


