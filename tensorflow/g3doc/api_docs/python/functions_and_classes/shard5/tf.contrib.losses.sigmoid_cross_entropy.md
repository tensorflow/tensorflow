### `tf.contrib.losses.sigmoid_cross_entropy(*args, **kwargs)` {#sigmoid_cross_entropy}

Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    multi_class_labels: [batch_size, num_classes] target labels in (0, 1).
    weights: Coefficients for the loss. The tensor must be a scalar, a tensor of
      shape [batch_size] or shape [batch_size, num_classes].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    weight: Deprecated alias for `weights`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weight` is invalid, or if
      `weight` is None.

