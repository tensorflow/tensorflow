Bijector constructed from custom callables.

Example Use:

```python
exp = Inline(
  forward_fn=tf.exp,
  inverse_fn=tf.log,
  inverse_log_det_jacobian_fn=(
    lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
  name="exp")
```

The above example is equivalent to the `Bijector` `Exp(event_ndims=1)`.
- - -

#### `tf.contrib.distributions.bijector.Inline.__init__(forward_fn=None, inverse_fn=None, inverse_log_det_jacobian_fn=None, forward_log_det_jacobian_fn=None, forward_event_shape_fn=None, forward_event_shape_tensor_fn=None, inverse_event_shape_fn=None, inverse_event_shape_tensor_fn=None, is_constant_jacobian=False, validate_args=False, name='inline')` {#Inline.__init__}

Creates a `Bijector` from callables.

##### Args:


*  <b>`forward_fn`</b>: Python callable implementing the forward transformation.
*  <b>`inverse_fn`</b>: Python callable implementing the inverse transformation.
*  <b>`inverse_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the inverse transformation.
*  <b>`forward_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the forward transformation.
*  <b>`forward_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`forward_event_shape_tensor_fn`</b>: Python callable implementing non-identical
    event shape changes. Default: shape is assumed unchanged.
*  <b>`inverse_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`inverse_event_shape_tensor_fn`</b>: Python callable implementing non-identical
    event shape changes. Default: shape is assumed unchanged.
*  <b>`is_constant_jacobian`</b>: `Boolean` indicating that the Jacobian is constant
    for all input arguments.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Inline.dtype` {#Inline.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Inline.event_ndims` {#Inline.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward(x, name='forward')` {#Inline.forward}

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

#### `tf.contrib.distributions.bijector.Inline.forward_event_shape(input_shape)` {#Inline.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Inline.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Inline.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.graph_parents` {#Inline.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse(y, name='inverse')` {#Inline.inverse}

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

#### `tf.contrib.distributions.bijector.Inline.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Inline.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.inverse_event_shape(output_shape)` {#Inline.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Inline.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Inline.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.is_constant_jacobian` {#Inline.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Inline.name` {#Inline.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Inline.validate_args` {#Inline.validate_args}

Returns True if Tensor arguments will be validated.


