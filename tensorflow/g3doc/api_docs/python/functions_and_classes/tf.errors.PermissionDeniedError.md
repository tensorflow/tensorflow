Raised when the caller does not have permission to run an operation.

For example, running the
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation could raise `PermissionDeniedError` if it receives the name of a
file for which the user does not have the read file permission.

- - -

#### `tf.errors.PermissionDeniedError.__init__(node_def, op, message)` {#PermissionDeniedError.__init__}

Creates a `PermissionDeniedError`.


