### `tf.global_variables()` {#global_variables}

Returns global variables.

Global variables are variables that are shared across machines in a
distributed environment. The `Variable()` constructor or `get_variable()`
automatically adds new variables to the graph collection
`GraphKeys.GLOBAL_VARIABLES`.
This convenience function returns the contents of that collection.

An alternative to global variables are local variables. See
[`tf.local_variables()`](../../api_docs/python/state_ops.md#local_variables)

##### Returns:

  A list of `Variable` objects.

