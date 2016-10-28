### `tf.contrib.losses.cosine_distance(*args, **kwargs)` {#cosine_distance}

Adds a cosine-distance loss to the training procedure. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`targets` is being deprecated, use `labels`. `weight` is being deprecated, use `weights`.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    predictions: An arbitrary matrix.
    labels: A `Tensor` whose shape matches 'predictions'
    dim: The dimension along which the cosine distance is computed.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    targets: Deprecated alias for `labels`.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `weights` is `None`.

