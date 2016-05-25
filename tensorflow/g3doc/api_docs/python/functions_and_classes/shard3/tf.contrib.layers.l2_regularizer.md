### `tf.contrib.layers.l2_regularizer(scale)` {#l2_regularizer}

Returns a function that can be used to apply L2 regularization to weights.

Small values of L2 can help prevent overfitting the training data.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

##### Returns:

  A function with signature `l2(weights, name=None)` that applies L2
  regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is outside of the range [0.0, 1.0] or if scale is not a
  float.

