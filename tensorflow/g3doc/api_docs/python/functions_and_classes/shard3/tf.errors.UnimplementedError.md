Raised when an operation has not been implemented.

Some operations may raise this error when passed otherwise-valid
arguments that it does not currently support. For example, running
the [`tf.nn.max_pool()`](../../api_docs/python/nn.md#max_pool) operation
would raise this error if pooling was requested on the batch dimension,
because this is not yet supported.

- - -

#### `tf.errors.UnimplementedError.__init__(node_def, op, message)` {#UnimplementedError.__init__}

Creates an `UnimplementedError`.


