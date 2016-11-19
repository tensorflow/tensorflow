### `tf.contrib.losses.sparse_softmax_cross_entropy(*args, **kwargs)` {#sparse_softmax_cross_entropy}

Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    labels: [batch_size, 1] or [batch_size] target labels of dtype `int32` or
      `int64` in the range `[0, num_classes)`.
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size] or [batch_size, 1].
    scope: the scope for the operations performed in computing the loss.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shapes of logits, labels, and weight are incompatible, or
      if `weight` is None.

