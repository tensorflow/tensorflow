### `tf.add_check_numerics_ops()` {#add_check_numerics_ops}

Connect a `check_numerics` to every floating point tensor.

`check_numerics` operations themselves are added for each `float` or `double`
tensor in the graph. For all ops in the graph, the `check_numerics` op for
all of its (`float` or `double`) inputs is guaranteed to run before the
`check_numerics` op on any of its outputs.

##### Returns:

  A `group` op depending on all `check_numerics` ops added.

