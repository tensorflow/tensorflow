### `tf.nn.elu(features, name=None)` {#elu}

Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.

