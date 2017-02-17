### `tf.nn.crelu(features, name=None)` {#crelu}

Computes Concatenated ReLU.

Concatenates a ReLU which selects only the positive part of the activation
with a ReLU which selects only the *negative* part of the activation.
Note that as a result this non-linearity doubles the depth of the activations.
Source: https://arxiv.org/abs/1603.05201

##### Args:


*  <b>`features`</b>: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
    `int16`, or `int8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with the same type as `features`.

