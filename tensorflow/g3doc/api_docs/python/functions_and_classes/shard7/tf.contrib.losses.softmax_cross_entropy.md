### `tf.contrib.losses.softmax_cross_entropy(*args, **kwargs)` {#softmax_cross_entropy}

Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits. (deprecated arguments) (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.softmax_cross_entropy instead.

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-11-25.
Instructions for updating:
`weight` is being deprecated, use `weights`

`weight` acts as a coefficient for the loss. If a scalar is provided,
then the loss is simply scaled by the given value. If `weight` is a
tensor of size [`batch_size`], then the loss weights apply to each
corresponding sample.

If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
    new_onehot_labels = onehot_labels * (1 - label_smoothing)
                        + label_smoothing / num_classes

##### Args:


*  <b>`logits`</b>: [batch_size, num_classes] logits outputs of the network .
*  <b>`onehot_labels`</b>: [batch_size, num_classes] target one_hot_encoded labels.
*  <b>`weights`</b>: Coefficients for the loss. The tensor must be a scalar or a tensor
    of shape [batch_size].
*  <b>`label_smoothing`</b>: If greater than 0 then smooth the labels.
*  <b>`scope`</b>: the scope for the operations performed in computing the loss.
*  <b>`weight`</b>: Deprecated alias for `weights`.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `logits` doesn't match that of `onehot_labels`
    or if the shape of `weight` is invalid or if `weight` is None.

