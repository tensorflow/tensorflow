Unknown error.

An example of where this error may be returned is if a Status value
received from another address space belongs to an error-space that
is not known to this address space. Also errors raised by APIs that
do not return enough error information may be converted to this
error.

- - -

#### `tf.errors.UnknownError.__init__(node_def, op, message, error_code=2)` {#UnknownError.__init__}

Creates an `UnknownError`.


