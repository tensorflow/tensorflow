Bijector which computes Y = g(X) = X.

Example Use:

```python
# Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
# ndim and 1 event ndim (i.e., vector of vectors).
identity = Identity(batch_ndims=1, event_ndims=1)
x = [[1., 2],
     [3, 4]]
x == identity.forward(x) == identity.inverse(x)
```
- - -

#### `tf.contrib.distributions.bijector.Identity.__init__(validate_args=False, name='Identity')` {#Identity.__init__}




- - -

#### `tf.contrib.distributions.bijector.Identity.dtype` {#Identity.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward(x, name='forward')` {#Identity.forward}

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

#### `tf.contrib.distributions.bijector.Identity.inverse(x, name='inverse')` {#Identity.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_and_inverse_log_det_jacobian(x, name='inverse_and_inverse_log_det_jacobian')` {#Identity.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_log_det_jacobian(x, name='inverse_log_det_jacobian')` {#Identity.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(x).

Mathematically, returns: log(det(dY/dX g^{-1}))(Y).

Note that forward_log_det_jacobian is the negative of this function. (See
is_constant_jacobian for related proof.)

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.is_constant_jacobian` {#Identity.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Identity.name` {#Identity.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Identity.parameters` {#Identity.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Identity.shaper` {#Identity.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Identity.validate_args` {#Identity.validate_args}

Returns True if Tensor arguments will be validated.


