### `tf.contrib.layers.l1_regularizer(scale)` {#l1_regularizer}

Returns a function that can be used to apply L1 regularization to weights.

L1 regularization encourages sparsity.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

##### Returns:

  A function with signature `l1(weights, name=None)` that apply L1
  regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is outside of the range [0.0, 1.0] or if scale is not a
  float.

