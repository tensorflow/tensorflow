Raised when an operation or step is cancelled.

For example, a long-running operation (e.g.
[`queue.enqueue()`](../../api_docs/python/io_ops.md#QueueBase.enqueue) may be
cancelled by running another operation (e.g.
[`queue.close(cancel_pending_enqueues=True)`](../../api_docs/python/io_ops.md#QueueBase.close),
or by [closing the session](../../api_docs/python/client.md#Session.close).
A step that is running such a long-running operation will fail by raising
`CancelledError`.

- - -

#### `tf.errors.CancelledError.__init__(node_def, op, message)` {#CancelledError.__init__}

Creates a `CancelledError`.


