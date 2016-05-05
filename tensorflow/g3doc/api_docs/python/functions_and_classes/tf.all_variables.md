### `tf.all_variables()` {#all_variables}

Returns all variables that must be saved/restored.

The `Variable()` constructor automatically adds new variables to the graph
collection `GraphKeys.VARIABLES`. This convenience function returns the
contents of that collection.

##### Returns:

  A list of `Variable` objects.

