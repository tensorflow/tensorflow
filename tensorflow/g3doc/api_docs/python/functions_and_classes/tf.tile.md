### `tf.tile(input, multiples, name=None)` {#tile}

Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. 1-D or higher.
*  <b>`multiples`</b>: A `Tensor` of type `int32`.
    1-D. Length must be the same as the number of dimensions in `input`
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.

