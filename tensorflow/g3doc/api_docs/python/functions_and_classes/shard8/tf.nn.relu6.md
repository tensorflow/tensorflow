### `tf.nn.relu6(features, name=None)` {#relu6}

Computes Rectified Linear 6: `min(max(features, 0), 6)`.

##### Args:


*  <b>`features`</b>: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
    `int16`, or `int8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with the same type as `features`.

