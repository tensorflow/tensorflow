Raised when unrecoverable data loss or corruption is encountered.

For example, this may be raised by running a
[`tf.WholeFileReader.read()`](../../api_docs/python/io_ops.md#WholeFileReader)
operation, if the file is truncated while it is being read.

- - -

#### `tf.errors.DataLossError.__init__(node_def, op, message)` {#DataLossError.__init__}

Creates a `DataLossError`.


