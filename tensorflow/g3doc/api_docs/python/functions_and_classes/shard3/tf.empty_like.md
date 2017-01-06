### `tf.empty_like(value, init=None)` {#empty_like}

Creates an empty Tensor with the same shape and type `dtype` as value.

The memory can optionally be initialized. This op is usually useful in
conjunction with in-place operations.

##### Args:


*  <b>`value`</b>: A `Tensor` whose shape will be used.
*  <b>`init`</b>: Initalize the returned tensor with the default value of
    `value.dtype()` if True.  Otherwise do not initialize.

##### Returns:


*  <b>`output`</b>: An empty Tensor of the specified shape and type.

