The operation was aborted, typically due to a concurrent action.

For example, running a
[`queue.enqueue()`](../../api_docs/python/io_ops.md#QueueBase.enqueue)
operation may raise `AbortedError` if a
[`queue.close()`](../../api_docs/python/io_ops.md#QueueBase.close) operation
previously ran.

- - -

#### `tf.errors.AbortedError.__init__(node_def, op, message)` {#AbortedError.__init__}

Creates an `AbortedError`.


