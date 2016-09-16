### `tf.contrib.losses.compute_weighted_loss(losses, weight=1.0)` {#compute_weighted_loss}

Computes the weighted loss.

##### Args:


*  <b>`losses`</b>: A tensor of size [batch_size, d1, ... dN].
*  <b>`weight`</b>: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.

##### Returns:

  A scalar `Tensor` that returns the weighted loss.

##### Raises:


*  <b>`ValueError`</b>: If the weight is None or the shape is not compatible with the
    losses shape or if the number of dimensions (rank) of either losses or
    weight is missing.

