Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

The [power transform](https://en.wikipedia.org/wiki/Power_transform) maps
inputs from `[0, inf]` to `[-1/c, inf]`; this is equivalent to the `inverse`
of this bijector.

This bijector is equivalent to the `Exp` bijector when `c=0`.
- - -

#### `tf.contrib.distributions.bijector.PowerTransform.__init__(power=0.0, event_ndims=0, validate_args=False, name='power_transform')` {#PowerTransform.__init__}

Instantiates the `PowerTransform` bijector.

##### Args:


*  <b>`power`</b>: Python `float` scalar indicating the transform power, i.e.,
    `Y = g(X) = (1 + X * c)**(1 / c)` where `c` is the `power`.
*  <b>`event_ndims`</b>: Python scalar indicating the number of dimensions associated
    with a particular draw from the distribution.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if `power < 0` or is not known statically.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.dtype` {#PowerTransform.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.event_ndims` {#PowerTransform.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward(x, name='forward')` {#PowerTransform.forward}

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

#### `tf.contrib.distributions.bijector.PowerTransform.forward_event_shape(input_shape)` {#PowerTransform.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#PowerTransform.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#PowerTransform.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.PowerTransform.graph_parents` {#PowerTransform.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse(y, name='inverse')` {#PowerTransform.inverse}

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

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#PowerTransform.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_event_shape(output_shape)` {#PowerTransform.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#PowerTransform.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#PowerTransform.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.PowerTransform.is_constant_jacobian` {#PowerTransform.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.name` {#PowerTransform.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.power` {#PowerTransform.power}

The `c` in: `Y = g(X) = (1 + X * c)**(1 / c)`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.validate_args` {#PowerTransform.validate_args}

Returns True if Tensor arguments will be validated.


