### `tf.train.limit_epochs(tensor, num_epochs=None, name=None)` {#limit_epochs}

Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

##### Args:


*  <b>`tensor`</b>: Any `Tensor`.
*  <b>`num_epochs`</b>: A positive integer (optional).  If specified, limits the number
    of steps the output tensor may be evaluated.
*  <b>`name`</b>: A name for the operations (optional).

##### Returns:

  tensor or `OutOfRange`.

##### Raises:


*  <b>`ValueError`</b>: if `num_epochs` is invalid.

