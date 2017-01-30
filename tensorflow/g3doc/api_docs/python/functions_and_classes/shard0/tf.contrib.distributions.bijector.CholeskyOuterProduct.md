Bijector which computes Y = g(X) = X X.T where X is a lower-triangular, positive-diagonal matrix.

`event_ndims` must be 0 or 2, i.e., scalar or matrix.

Note: the upper-triangular part of X is ignored (whether or not its zero).

Examples:

```python
bijector.CholeskyOuterProduct(event_ndims=2).forward(x=[[1., 0], [2, 1]])
# Result: [[1, 1], [1, 5]], i.e., x x.T

bijector.SoftmaxCentered(event_ndims=2).inverse(y=[[1., 1], [1, 5]])
# Result: [[1, 0], [2, 1]], i.e., chol(y).
```
- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.__init__(event_ndims=2, validate_args=False, name='cholesky_outer_product')` {#CholeskyOuterProduct.__init__}

Instantiates the `CholeskyOuterProduct` bijector.

##### Args:


*  <b>`event_ndims`</b>: `constant` `int32` scalar `Tensor` indicating the number of
    dimensions associated with a particular draw from the distribution. Must
    be 0 or 2.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if event_ndims is neither 0 or 2.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.dtype` {#CholeskyOuterProduct.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward(x, name='forward', **condition_kwargs)` {#CholeskyOuterProduct.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_event_shape(input_shape, name='forward_event_shape')` {#CholeskyOuterProduct.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.get_forward_event_shape(input_shape)` {#CholeskyOuterProduct.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.get_inverse_event_shape(output_shape)` {#CholeskyOuterProduct.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.graph_parents` {#CholeskyOuterProduct.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse(y, name='inverse', **condition_kwargs)` {#CholeskyOuterProduct.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_event_shape(output_shape, name='inverse_event_shape')` {#CholeskyOuterProduct.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.is_constant_jacobian` {#CholeskyOuterProduct.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.name` {#CholeskyOuterProduct.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.shaper` {#CholeskyOuterProduct.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.validate_args` {#CholeskyOuterProduct.validate_args}

Returns True if Tensor arguments will be validated.


