### `tf.contrib.losses.hinge_loss(*args, **kwargs)` {#hinge_loss}

Method that returns the loss tensor for hinge loss. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.hinge_loss instead.

##### Args:


*  <b>`logits`</b>: The logits, a float tensor.
*  <b>`labels`</b>: The ground truth output tensor. Its shape should match the shape of
    logits. The values of the tensor are expected to be 0.0 or 1.0.
*  <b>`scope`</b>: The scope for the operations performed in computing the loss.

##### Returns:

  A `Tensor` of same shape as `logits` and `labels` representing the loss
    values across the batch.

##### Raises:


*  <b>`ValueError`</b>: If the shapes of `logits` and `labels` don't match.

