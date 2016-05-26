Raised when an operation iterates past the valid input range.

This exception is raised in "end-of-file" conditions, such as when a
[`queue.dequeue()`](../../api_docs/python/io_ops.md#QueueBase.dequeue)
operation is blocked on an empty queue, and a
[`queue.close()`](../../api_docs/python/io_ops.md#QueueBase.close)
operation executes.

- - -

#### `tf.errors.OutOfRangeError.__init__(node_def, op, message)` {#OutOfRangeError.__init__}

Creates an `OutOfRangeError`.


