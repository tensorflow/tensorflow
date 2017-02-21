### `tf.broadcast_static_shape(shape_x, shape_y)` {#broadcast_static_shape}

Returns the broadcasted static shape between `shape_x` and `shape_y`.

##### Args:


*  <b>`shape_x`</b>: A `TensorShape`
*  <b>`shape_y`</b>: A `TensorShape`

##### Returns:

  A `TensorShape` representing the broadcasted shape.

##### Raises:


*  <b>`ValueError`</b>: If the two shapes can not be broadcasted.

