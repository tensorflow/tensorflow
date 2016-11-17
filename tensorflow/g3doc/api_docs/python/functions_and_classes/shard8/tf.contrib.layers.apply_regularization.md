### `tf.contrib.layers.apply_regularization(regularizer, weights_list=None)` {#apply_regularization}

Returns the summed penalty by applying `regularizer` to the `weights_list`.

Adding a regularization penalty over the layer weights and embedding weights
can help prevent overfitting the training data. Regularization over layer
biases is less common/useful, but assuming proper data preprocessing/mean
subtraction, it usually shouldn't hurt much either.

##### Args:


*  <b>`regularizer`</b>: A function that takes a single `Output` argument and returns
    a scalar `Output` output.
*  <b>`weights_list`</b>: List of weights `Output`s or `Variables` to apply
    `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
    `None`.

##### Returns:

  A scalar representing the overall regularization penalty.

##### Raises:


*  <b>`ValueError`</b>: If `regularizer` does not return a scalar output, or if we find
      no weights.

