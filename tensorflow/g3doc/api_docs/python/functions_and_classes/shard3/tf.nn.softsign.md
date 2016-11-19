### `tf.nn.softsign(features, name=None)` {#softsign}

Computes softsign: `features / (abs(features) + 1)`.

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.

