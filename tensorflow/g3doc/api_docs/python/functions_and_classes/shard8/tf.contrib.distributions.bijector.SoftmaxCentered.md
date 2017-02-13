Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
bijection, the forward transformation appends a value to the input and the
inverse removes this coordinate.  The appended coordinate represents a pivot,
e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
coordinate.

Because we append a coordinate, this bijector only supports `event_ndim in [0,
1]`, i.e., scalars and vectors.

Example Use:

```python
bijector.SoftmaxCentered(event_ndims=1).forward(tf.log([2, 3, 4]))
# Result: [0.2, 0.3, 0.4, 0.1]
# Extra result: 0.1

bijector.SoftmaxCentered(event_ndims=1).inverse([0.2, 0.3, 0.4, 0.1])
# Result: tf.log([2, 3, 4])
# Extra coordinate removed.
```

At first blush it may seem like the [Invariance of domain](
https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
implementation is not a bijection.  However, the appended dimension
makes the (forward) image non-open and the theorem does not directly apply.
- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.__init__(event_ndims=0, validate_args=False, name='softmax_centered')` {#SoftmaxCentered.__init__}




- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.dtype` {#SoftmaxCentered.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.event_ndims` {#SoftmaxCentered.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward(x, name='forward')` {#SoftmaxCentered.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_event_shape(input_shape)` {#SoftmaxCentered.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#SoftmaxCentered.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#SoftmaxCentered.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.graph_parents` {#SoftmaxCentered.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse(y, name='inverse')` {#SoftmaxCentered.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#SoftmaxCentered.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_event_shape(output_shape)` {#SoftmaxCentered.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#SoftmaxCentered.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#SoftmaxCentered.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.is_constant_jacobian` {#SoftmaxCentered.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.name` {#SoftmaxCentered.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.validate_args` {#SoftmaxCentered.validate_args}

Returns True if Tensor arguments will be validated.


