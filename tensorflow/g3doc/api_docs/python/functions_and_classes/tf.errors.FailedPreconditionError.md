Operation was rejected because the system is not in a state to execute it.

This exception is most commonly raised when running an operation
that reads a [`tf.Variable`](../../api_docs/python/state_ops.md#Variable)
before it has been initialized.

- - -

#### `tf.errors.FailedPreconditionError.__init__(node_def, op, message)` {#FailedPreconditionError.__init__}

Creates a `FailedPreconditionError`.


