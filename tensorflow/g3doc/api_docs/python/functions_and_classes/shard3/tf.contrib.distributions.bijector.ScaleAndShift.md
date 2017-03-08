Bijector which computes Y = g(X; shift, scale) = matmul(scale, X) + shift.

`scale` is either a non-zero scalar, or a lower triangular matrix with
non-zero diagonal.  This means the `Bijector` will be invertible and
computation of determinant and inverse will be efficient.

As a result, the mean and covariance are transformed:

```
E[Y] = matmul(scale, E[X])
Cov[Y] = matmul(scale, matmul(Cov[X], scale, transpose_b=True))
```

Example Use:

```python
# No batch, scalar
mu = 0     # shape=[]
sigma = 1  # shape=[], treated like a 1x1 matrix.
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 0
# b.shaper.event_ndims == 0

# One batch, scalar.
mu = ...    # shape=[b], b>0
sigma = ... # shape=[b], b>0, treated like a batch of 1x1 matrices
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 1
# b.shaper.event_ndims == 0

# No batch, multivariate.
mu = ...    # shape=[d],    d>0
sigma = ... # shape=[d, d], d>0, treated like a single dxd matrix.
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
# b.shaper.batch_ndims == 0
# b.shaper.event_ndims == 1

# (B1*B2*...*Bb)-batch, multivariate.
mu = ...    # shape=[B1,...,Bb, d],    b>0, d>0
sigma = ... # shape=[B1,...,Bb, d, d], b>0, d>0
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
# b.shaper.batch_ndims == b
# b.shaper.event_ndims == 1

# Mu is broadcast:
mu = 1
sigma = [I, I]  # I is a 3x3 identity matrix.
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
x = numpy.ones(S + sigma.shape)
b.forward(x) # == x + 1
```
- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.__init__(shift, scale, event_ndims=0, validate_args=False, name='scale_and_shift')` {#ScaleAndShift.__init__}

Instantiates the `ScaleAndShift` bijector.

This `Bijector` is initialized with `scale` and `shift` `Tensors`, giving
the forward operation:

```Y = g(X) = matmul(scale, X) + shift```

##### Args:


*  <b>`shift`</b>: Numeric `Tensor`.
*  <b>`scale`</b>: Numeric `Tensor` of same `dtype` as `shift`.  If `event_ndims = 0`,
    `scale` is treated like a `1x1` matrix or a batch thereof.
    Otherwise, the last two dimensions of `scale` define a matrix.
    `scale` must have non-negative diagonal entries.  The upper triangular
    part of `scale` is ignored, effectively making it lower triangular.
*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution.  Must be 0 or 1
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.dtype` {#ScaleAndShift.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward(x, name='forward', **condition_kwargs)` {#ScaleAndShift.forward}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward_event_shape(input_shape, name='forward_event_shape')` {#ScaleAndShift.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.get_forward_event_shape(input_shape)` {#ScaleAndShift.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.get_inverse_event_shape(output_shape)` {#ScaleAndShift.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse(y, name='inverse', **condition_kwargs)` {#ScaleAndShift.inverse}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_event_shape(output_shape, name='inverse_event_shape')` {#ScaleAndShift.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.is_constant_jacobian` {#ScaleAndShift.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.name` {#ScaleAndShift.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.parameters` {#ScaleAndShift.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.scale` {#ScaleAndShift.scale}




- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.shaper` {#ScaleAndShift.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.shift` {#ScaleAndShift.shift}




- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.validate_args` {#ScaleAndShift.validate_args}

Returns True if Tensor arguments will be validated.


