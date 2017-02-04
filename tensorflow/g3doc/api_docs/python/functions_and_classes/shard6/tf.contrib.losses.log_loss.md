### `tf.contrib.losses.log_loss(*args, **kwargs)` {#log_loss}

Adds a Log Loss term to the training procedure. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.log_loss instead.

`weights` acts as a coefficient for the loss. If a scalar is provided, then
the loss is simply scaled by the given value. If `weights` is a tensor of size
[batch_size], then the total loss for each sample of the batch is rescaled
by the corresponding element in the `weights` vector. If the shape of
`weights` matches the shape of `predictions`, then the loss of each
measurable element of `predictions` is scaled by the corresponding value of
`weights`.

##### Args:


*  <b>`predictions`</b>: The predicted outputs.
*  <b>`labels`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`weights`</b>: Coefficients for the loss a scalar, a tensor of shape
    [batch_size] or a tensor whose shape matches `predictions`.
*  <b>`epsilon`</b>: A small increment to add to avoid taking a log of zero.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` representing the loss value.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` doesn't match that of `labels` or
    if the shape of `weights` is invalid.

