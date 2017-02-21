### `tf.local_variables()` {#local_variables}

Returns local variables.

Local variables - per process variables, usually not saved/restored to
checkpoint and used for temporary or intermediate values.
For example, they can be used as counters for metrics computation or
number of epochs this machine has read data.
The `tf.contrib.framework.local_variable()` function automatically adds the
new variable to `GraphKeys.LOCAL_VARIABLES`.
This convenience function returns the contents of that collection.

An alternative to local variables are global variables. See
[`tf.global_variables()`](../../api_docs/python/state_ops.md#global_variables)

##### Returns:

  A list of local `Variable` objects.

