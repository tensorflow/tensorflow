### `tf.empty(output_shape, dtype, init=False)` {#empty}

Creates an empty Tensor with shape `output_shape` and type `dtype`.

The memory can optionally be initialized. This is usually useful in
conjunction with in-place operations.

##### Args:


*  <b>`output_shape`</b>: 1-D `Tensor` indicating the shape of the output.
*  <b>`dtype`</b>: The element type of the returned tensor.
*  <b>`init`</b>: `bool` indicating whether or not to zero the allocated memory.

##### Returns:


*  <b>`output`</b>: An empty Tensor of the specified type.

