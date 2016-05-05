Raised when a requested entity (e.g., a file or directory) was not found.

For example, running the
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation could raise `NotFoundError` if it receives the name of a file that
does not exist.

- - -

#### `tf.errors.NotFoundError.__init__(node_def, op, message)` {#NotFoundError.__init__}

Creates a `NotFoundError`.


