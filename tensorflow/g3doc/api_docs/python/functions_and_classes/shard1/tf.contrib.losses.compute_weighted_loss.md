### `tf.contrib.losses.compute_weighted_loss(*args, **kwargs)` {#compute_weighted_loss}

Computes the weighted loss. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.

