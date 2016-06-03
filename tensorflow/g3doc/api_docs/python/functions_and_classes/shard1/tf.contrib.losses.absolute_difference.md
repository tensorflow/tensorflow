### `tf.contrib.losses.absolute_difference(predictions, targets, weight=1.0, scope=None)` {#absolute_difference}

Adds an Absolute Difference loss to the training procedure.

`weight` acts as a coefficient for the loss. If a scalar is provided, then the
loss is simply scaled by the given value. If `weight` is a tensor of size
[batch_size], then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weight` vector. If the shape of
`weight` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weight`.

##### Args:


*  <b>`predictions`</b>: The predicted outputs.
*  <b>`targets`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`weight`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `targets` or
    if the shape of `weight` is invalid.

