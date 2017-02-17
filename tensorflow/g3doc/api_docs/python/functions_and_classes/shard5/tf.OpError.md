A generic error that is raised when TensorFlow execution fails.

Whenever possible, the session will raise a more specific subclass
of `OpError` from the `tf.errors` module.
- - -

#### `tf.OpError.__init__(node_def, op, message, error_code)` {#OpError.__init__}

Creates a new `OpError` indicating that a particular op failed.

##### Args:


*  <b>`node_def`</b>: The `node_def_pb2.NodeDef` proto representing the op that
    failed, if known; otherwise None.
*  <b>`op`</b>: The `ops.Operation` that failed, if known; otherwise None.
*  <b>`message`</b>: The message string describing the failure.
*  <b>`error_code`</b>: The `error_codes_pb2.Code` describing the error.


- - -

#### `tf.OpError.__str__()` {#OpError.__str__}




- - -

#### `tf.OpError.error_code` {#OpError.error_code}

The integer error code that describes the error.


- - -

#### `tf.OpError.message` {#OpError.message}

The error message that describes the error.


- - -

#### `tf.OpError.node_def` {#OpError.node_def}

The `NodeDef` proto representing the op that failed.


- - -

#### `tf.OpError.op` {#OpError.op}

The operation that failed, if known.

*N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
or `Recv` op, there will be no corresponding
[`Operation`](../../api_docs/python/framework.md#Operation)
object.  In that case, this will return `None`, and you should
instead use the [`OpError.node_def`](#OpError.node_def) to
discover information about the op.

##### Returns:

  The `Operation` that failed, or None.


