Bijector which computes Y = g(X; shift, scale) = matmul(scale, X) + shift.

`shift` is a numeric `Tensor`, while `scale` is constructed from several
arguments specified in the constructor.

The most general `scale` construction is described as follows:

```python
scale = (
  scale_identity_multiplier * tf.diag(tf.ones(d)) +
  tf.diag(scale_diag) +
  scale_tril +
  scale_perturb_factor @ diag(scale_perturb_diag) @
    tf.transpose([scale_perturb_factor])

scale = c * I + diag(D1) + V @ diag(D2) @ V^T + tril(L)
```

Example Use:

```python
# No batch, scalar
mu = 0     # shape=[]
sigma = 1. # shape=[], treated like a 1x1 matrix.
# Corresponds to forward: x + mu.
b = Affine(shift=mu)

# No batch, scalar
mu = 0     # shape=[]
sigma = 3.  # shape=[], treated like a 1x1 matrix.
# Corresponds to forward: 3 * x + mu.
b = Affine(
  shift=mu,
  scale_identity_multiplier=3.0)

# One batch, scalar.
mu = ...    # shape=[b], b>0
sigma = 2. # shape=[], b>0, treated like a batch of 1x1 matrices
# Corresponds to forward: 2 * x + mu.
b = Affine(
  shift=mu,
  scale_identity_multiplier=2.0)

# No batch, multivariate.
mu = [1., 2, 3]  # shape=[3],
diag = [1, 3, 3] # shape=[3, 3], treated like 3x3 matrix.
b = Affine(
  shift=mu,
  scale_identity_multiplier=None,
  scale_diag=diag,
  event_ndims=1)

# Low rank update.
mu = [1, 2, 3]    # shape=[3],
d2 = [2, 1] # shape=[2], treated like a 2x2 matrix.
v = [[1, 0] [0, 1], [0, 0]] # shape=[2, 3]
d1 = [1, 3, 3] # shape=[3, 3], treated like 3x3 matrix.
# Corresponds to scale of the form d1 + v * d2 * v^T
b = Affine(
  shift=mu,
  scale_identity_multiplier=None,
  scale_diag=d1,
  scale_perturb_diag=d2,
  scale_perturb_factor=v,
  event_ndims=1)

```
- - -

#### `tf.contrib.distributions.bijector.Affine.__init__(shift, scale_identity_multiplier=None, scale_diag=None, scale_tril=None, scale_perturb_diag=None, scale_perturb_factor=None, event_ndims=0, validate_args=False, name='affine')` {#Affine.__init__}

Instantiates the `Affine` bijector.

This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
giving the forward operation:

```Y = g(X) = scale @ X + shift```

##### Args:


*  <b>`shift`</b>: Numeric `Tensor`.
*  <b>`scale_identity_multiplier`</b>: floating point rank 0 `Tensor` representing a
    scaling done to the identity matrix.
    The default is 1.0.  If this is set to `None`, do not scale by an
    identity matrix.
*  <b>`scale_diag`</b>: Numeric `Tensor` representing the diagonal matrix.
    `scale_diag` has shape [N1, N2, ... k], which represents a k x k
    diagonal matrix.
    The default is `None`. If this is set to `None`, scale_diag is not used
    for scale construction.
*  <b>`scale_tril`</b>: Numeric `Tensor` representing the diagonal matrix.
    `scale_diag` has shape [N1, N2, ... k, k], which represents a k x k
    lower triangular matrix.
    The default is `None`. If this is set to `None`, scale_tril is not used
    for scale construction.
*  <b>`scale_perturb_diag`</b>: Numeric `Tensor` representing the diagonal matrix.
    `scale_perturb_diag` has shape [N1, N2, ... r], which represents an
    r x r Diagonal matrix.
    The default is`None`. If this is set to `None`, low rank updates will
    take the form `scale_perturb_factor * scale_perturb_factor^T`.
*  <b>`scale_perturb_factor`</b>: Numeric `Tensor` representing factor matrix with
    last two dimensions of shape `(k, r)`.
    The default is `None`. If this is set to `None`, no rank update is
    performed.
*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution. Must be 0 or 1.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Affine.dtype` {#Affine.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward(x, name='forward', **condition_kwargs)` {#Affine.forward}

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

#### `tf.contrib.distributions.bijector.Affine.forward_event_shape(input_shape, name='forward_event_shape')` {#Affine.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Affine.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Affine.get_forward_event_shape(input_shape)` {#Affine.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Affine.get_inverse_event_shape(output_shape)` {#Affine.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Affine.graph_parents` {#Affine.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse(y, name='inverse', **condition_kwargs)` {#Affine.inverse}

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

#### `tf.contrib.distributions.bijector.Affine.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Affine.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Affine.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Affine.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Affine.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Affine.is_constant_jacobian` {#Affine.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Affine.name` {#Affine.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Affine.scale` {#Affine.scale}




- - -

#### `tf.contrib.distributions.bijector.Affine.shaper` {#Affine.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Affine.shift` {#Affine.shift}




- - -

#### `tf.contrib.distributions.bijector.Affine.validate_args` {#Affine.validate_args}

Returns True if Tensor arguments will be validated.


