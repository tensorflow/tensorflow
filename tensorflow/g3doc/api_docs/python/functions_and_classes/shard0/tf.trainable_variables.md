### `tf.trainable_variables()` {#trainable_variables}

Returns all variables created with `trainable=True`.

When passed `trainable=True`, the `Variable()` constructor automatically
adds new variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
contents of that collection.

##### Returns:

  A list of Variable objects.

