### `tf.contrib.losses.hinge_loss(*args, **kwargs)` {#hinge_loss}

Method that returns the loss tensor for hinge loss. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`target` is being deprecated, use `labels`.

  Args:
    logits: The logits, a float tensor.
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    scope: The scope for the operations performed in computing the loss.
    target: Deprecated alias for `labels`.

  Returns:
    A `Tensor` of same shape as logits and target representing the loss values
      across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match.

