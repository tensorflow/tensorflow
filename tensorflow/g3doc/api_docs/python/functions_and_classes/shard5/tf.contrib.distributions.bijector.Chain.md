Bijector which applies a sequence of bijectors.

Example Use:

```python
chain = Chain([Exp(), Softplus()], name="one_plus_exp")
```

Results in:

* Forward:

 ```python
 exp = Exp()
 softplus = Softplus()
 Chain([exp, softplus]).forward(x)
 = exp.forward(softplus.forward(x))
 = tf.exp(tf.log(1. + tf.exp(x)))
 = 1. + tf.exp(x)
 ```

* Inverse:

 ```python
 exp = Exp()
 softplus = Softplus()
 Chain([exp, softplus]).inverse(y)
 = softplus.inverse(exp.inverse(y))
 = tf.log(tf.exp(tf.log(y)) - 1.)
 = tf.log(y - 1.)
 ```
- - -

#### `tf.contrib.distributions.bijector.Chain.__init__(bijectors=(), validate_args=False, name=None)` {#Chain.__init__}

Instantiates `Chain` bijector.

##### Args:


*  <b>`bijectors`</b>: Python list of bijector instances. An empty list makes this
    bijector equivalent to the `Identity` bijector.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str`, name given to ops managed by this object. Default:
    E.g., `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

##### Raises:


*  <b>`ValueError`</b>: if bijectors have different dtypes.


- - -

#### `tf.contrib.distributions.bijector.Chain.bijectors` {#Chain.bijectors}




- - -

#### `tf.contrib.distributions.bijector.Chain.dtype` {#Chain.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Chain.event_ndims` {#Chain.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward(x, name='forward')` {#Chain.forward}

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

#### `tf.contrib.distributions.bijector.Chain.forward_event_shape(input_shape)` {#Chain.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Chain.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Chain.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.graph_parents` {#Chain.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse(y, name='inverse')` {#Chain.inverse}

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

#### `tf.contrib.distributions.bijector.Chain.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Chain.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.inverse_event_shape(output_shape)` {#Chain.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Chain.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Chain.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.is_constant_jacobian` {#Chain.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Chain.name` {#Chain.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Chain.validate_args` {#Chain.validate_args}

Returns True if Tensor arguments will be validated.


