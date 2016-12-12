### `tf.contrib.losses.compute_weighted_loss(*args, **kwargs)` {#compute_weighted_loss}

Computes the weighted loss. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.compute_weighted_loss instead.

##### Args:


*  <b>`losses`</b>: A tensor of size [batch_size, d1, ... dN].
*  <b>`weights`</b>: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
*  <b>`scope`</b>: the scope for the operations performed in computing the loss.

##### Returns:

  A scalar `Tensor` that returns the weighted loss.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is `None` or the shape is not compatible with
    `losses`, or if the number of dimensions (rank) of either `losses` or
    `weights` is missing.

