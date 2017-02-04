### `tf.contrib.losses.sigmoid_cross_entropy(*args, **kwargs)` {#sigmoid_cross_entropy}

Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.sigmoid_cross_entropy instead.

`weights` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weights` is a
tensor of size [`batch_size`], then the loss weights apply to each
corresponding sample.

If `label_smoothing` is nonzero, smooth the labels towards 1/2:

    new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                            + 0.5 * label_smoothing

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`multi_class_labels`</b>: [batch_size, num_classes] labels in (0, 1).
*  <b>`weights`</b>: Coefficients for the loss. The tensor must be a scalar, a tensor of
    shape [batch_size] or shape [batch_size, num_classes].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `logits` doesn't match that of
    `multi_class_labels` or if the shape of `weights` is invalid, or if
    `weights` is None.

