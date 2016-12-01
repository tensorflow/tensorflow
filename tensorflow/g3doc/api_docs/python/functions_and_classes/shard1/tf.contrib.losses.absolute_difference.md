### `tf.contrib.losses.absolute_difference(*args, **kwargs)` {#absolute_difference}

Adds an Absolute Difference loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weight` is invalid.

