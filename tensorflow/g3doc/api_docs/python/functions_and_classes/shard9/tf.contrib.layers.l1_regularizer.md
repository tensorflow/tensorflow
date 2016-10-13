### `tf.contrib.layers.l1_regularizer(scale, scope=None)` {#l1_regularizer}

Returns a function that can be used to apply L1 regularization to weights.

L1 regularization encourages sparsity.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
*  <b>`scope`</b>: An optional scope name.

##### Returns:

  A function with signature `l1(weights)` that apply L1 regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is negative or if scale is not a float.

