### `tf.clip_by_norm(t, clip_norm, axes=None, name=None)` {#clip_by_norm}

Clips tensor values to a maximum L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
along the dimensions given in `axes`. Specifically, in the default case
where all dimensions are used for calculation, if the L2-norm of `t` is
already less than or equal to `clip_norm`, then `t` is not modified. If
the L2-norm is greater than `clip_norm`, then this operation returns a
tensor of the same type and shape as `t` with its values set to:

`t * clip_norm / l2norm(t)`

In this case, the L2-norm of the output tensor is `clip_norm`.

As another example, if `t` is a matrix and `axes == [1]`, then each row
of the output will have L2-norm equal to `clip_norm`. If `axes == [0]`
instead, each column of the output will be clipped.

This operation is typically used to clip gradients before applying them with
an optimizer.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
*  <b>`axes`</b>: A 1-D (vector) `Tensor` of type int32 containing the dimensions
    to use for computing the L2-norm. If `None` (the default), uses all
    dimensions.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A clipped `Tensor`.

