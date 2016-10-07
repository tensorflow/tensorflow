Raised when an entity that we attempted to create already exists.

For example, running an operation that saves a file
(e.g. [`tf.train.Saver.save()`](../../api_docs/python/train.md#Saver.save))
could potentially raise this exception if an explicit filename for an
existing file was passed.

- - -

#### `tf.errors.AlreadyExistsError.__init__(node_def, op, message)` {#AlreadyExistsError.__init__}

Creates an `AlreadyExistsError`.


