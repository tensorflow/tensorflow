Bijector which computes Y = g(X; shift, scale) = scale * X + shift.

Example Use:

```python
# No batch, scalar.
mu = 0     # shape=[]
sigma = 1  # shape=[]
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 0
# b.shaper.event_ndims == 0

# One batch, scalar.
mu = ...    # shape=[b], b>0
sigma = ... # shape=[b], b>0
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 1
# b.shaper.event_ndims == 0

# No batch, multivariate.
mu = ...    # shape=[d],    d>0
sigma = ... # shape=[d, d], d>0
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

Instantiates the `Exp` bijector.

##### Args:


*  <b>`shift`</b>: `Tensor` used to shift input, i.e., `Y = g(X) = scale * X + shift`.
*  <b>`scale`</b>: `Tensor` used to scale input, i.e., `Y = g(X) = scale * X + shift`.
*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution.
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


