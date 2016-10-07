### `tf.contrib.losses.sum_of_pairwise_squares(*args, **kwargs)` {#sum_of_pairwise_squares}

Adds a pairwise-errors-squared loss to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-01.
Instructions for updating:
Use mean_pairwise_squared_error.

  Unlike the sum_of_squares loss, which is a measure of the differences between
  corresponding elements of `predictions` and `targets`, sum_of_pairwise_squares
  is a measure of the differences between pairs of corresponding elements of
  `predictions` and `targets`.

  For example, if `targets`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of size [batch_size, d0, ... dN], the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector.

  Args:
    predictions: The predicted outputs, a tensor of size [batch_size, d0, .. dN]
      where N+1 is the total number of dimensions in `predictions`.
    targets: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weight: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.

