Distribution that supports intrinsic parameters (local latents).

Subclasses of this distribution may have additional keyword arguments passed
to their sample-based methods (i.e. `sample`, `log_prob`, etc.).
- - -

#### `tf.contrib.distributions.ConditionalDistribution.__init__(dtype, is_continuous, reparameterization_type, validate_args, allow_nan_stats, parameters=None, graph_parents=None, name=None)` {#ConditionalDistribution.__init__}

Constructs the `Distribution`.

**This is a private method for subclass use.**

##### Args:


*  <b>`dtype`</b>: The type of the event samples. `None` implies no type-enforcement.
*  <b>`is_continuous`</b>: Python boolean. If `True` this
    `Distribution` is continuous over its supported domain.
*  <b>`reparameterization_type`</b>: Instance of `ReparameterizationType`.
    If `distributions.FULLY_REPARAMETERIZED`, this
    `Distribution` can be reparameterized in terms of some standard
    distribution with a function whose Jacobian is constant for the support
    of the standard distribution.  If `distributions.NOT_REPARAMETERIZED`,
    then no such reparameterization is available.
*  <b>`validate_args`</b>: Python boolean.  Whether to validate input with asserts.
    If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Python boolean.  If `False`, raise an
    exception if a statistic (e.g., mean, mode) is undefined for any batch
    member. If True, batch members with valid parameters leading to
    undefined statistics will return `NaN` for this statistic.
*  <b>`parameters`</b>: Python dictionary of parameters used to instantiate this
    `Distribution`.
*  <b>`graph_parents`</b>: Python list of graph prerequisites of this `Distribution`.
*  <b>`name`</b>: A name for this distribution. Default: subclass name.

##### Raises:


*  <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.allow_nan_stats` {#ConditionalDistribution.allow_nan_stats}

Python boolean describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense.  E.g., the variance
of a Cauchy distribution is infinity.  However, sometimes the
statistic is undefined, e.g., if a distribution's pdf does not achieve a
maximum within the support of the distribution, the mode is undefined.
If the mean is undefined, then by definition the variance is undefined.
E.g. the mean for Student's T for df = 1 is undefined (no clear way to say
it is either + or - infinity), so the variance = E[(X - mean)^2] is also
undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python boolean.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.batch_shape` {#ConditionalDistribution.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.batch_shape_tensor(name='batch_shape_tensor')` {#ConditionalDistribution.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.cdf(*args, **kwargs)` {#ConditionalDistribution.cdf}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.copy(**override_parameters_kwargs)` {#ConditionalDistribution.copy}

Creates a deep copy of the distribution.

Note: the copy distribution may continue to depend on the original
intialization arguments.

##### Args:


*  <b>`**override_parameters_kwargs`</b>: String/value dictionary of initialization
    arguments to override with new values.

##### Returns:


*  <b>`distribution`</b>: A new instance of `type(self)` intitialized from the union
    of self.parameters and override_parameters_kwargs, i.e.,
    `dict(self.parameters, **override_parameters_kwargs)`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.covariance(name='covariance')` {#ConditionalDistribution.covariance}

Covariance.

Covariance is (possibly) defined only for non-scalar-event distributions.

For example, for a length-`k`, vector-valued distribution, it is calculated
as,

```none
Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
```

where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
denotes expectation.

Alternatively, for non-vector, multivariate distributions (e.g.,
matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
under some vectorization of the events, i.e.,

```none
Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
````

where `Cov` is a (batch of) `k' x k'` matrices,
`0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
mapping indices of this distribution's event dimensions to indices of a
length-`k'` vector.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`covariance`</b>: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
    where the first `n` dimensions are batch coordinates and
    `k' = reduce_prod(self.event_shape)`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.dtype` {#ConditionalDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.entropy(name='entropy')` {#ConditionalDistribution.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.event_shape` {#ConditionalDistribution.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.event_shape_tensor(name='event_shape_tensor')` {#ConditionalDistribution.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.is_continuous` {#ConditionalDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.ConditionalDistribution.is_scalar_batch(name='is_scalar_batch')` {#ConditionalDistribution.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.is_scalar_event(name='is_scalar_event')` {#ConditionalDistribution.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.log_cdf(*args, **kwargs)` {#ConditionalDistribution.log_cdf}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.log_prob(*args, **kwargs)` {#ConditionalDistribution.log_prob}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.log_survival_function(*args, **kwargs)` {#ConditionalDistribution.log_survival_function}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.mean(name='mean')` {#ConditionalDistribution.mean}

Mean.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.mode(name='mode')` {#ConditionalDistribution.mode}

Mode.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.name` {#ConditionalDistribution.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#ConditionalDistribution.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.

Subclasses should override class method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.param_static_shapes(cls, sample_shape)` {#ConditionalDistribution.param_static_shapes}

param_shapes with static (i.e. `TensorShape`) shapes.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.  Assumes that
the sample's shape is known statically.

Subclasses should override class method `_param_shapes` to return
constant-valued tensors when constant values are fed.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.parameters` {#ConditionalDistribution.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.prob(*args, **kwargs)` {#ConditionalDistribution.prob}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.reparameterization_type` {#ConditionalDistribution.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.sample(*args, **kwargs)` {#ConditionalDistribution.sample}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.stddev(name='stddev')` {#ConditionalDistribution.stddev}

Standard deviation.

Standard deviation is defined as,

```none
stddev = E[(X - E[X])**2]**0.5
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `stddev.shape = batch_shape + event_shape`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`stddev`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.survival_function(*args, **kwargs)` {#ConditionalDistribution.survival_function}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.validate_args` {#ConditionalDistribution.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.ConditionalDistribution.variance(name='variance')` {#ConditionalDistribution.variance}

Variance.

Variance is defined as,

```none
Var = E[(X - E[X])**2]
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `Var.shape = batch_shape + event_shape`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`variance`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.


