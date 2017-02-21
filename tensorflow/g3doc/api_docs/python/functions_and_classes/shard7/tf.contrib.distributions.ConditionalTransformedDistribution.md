A TransformedDistribution that allows intrinsic conditioning.
- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.__init__(distribution, bijector=None, batch_shape=None, event_shape=None, validate_args=False, name=None)` {#ConditionalTransformedDistribution.__init__}

Construct a Transformed Distribution.

##### Args:


*  <b>`distribution`</b>: The base distribution instance to transform. Typically an
    instance of `Distribution`.
*  <b>`bijector`</b>: The object responsible for calculating the transformation.
    Typically an instance of `Bijector`. `None` means `Identity()`.
*  <b>`batch_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
    `batch_shape`; valid only if `distribution.is_scalar_batch()`.
*  <b>`event_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
    `event_shape`; valid only if `distribution.is_scalar_event()`.
*  <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
*  <b>`name`</b>: Python `str` name prefixed to Ops created by this class. Default:
    `bijector.name + distribution.name`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.allow_nan_stats` {#ConditionalTransformedDistribution.allow_nan_stats}

Python `bool` describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense. E.g., the variance of a
Cauchy distribution is infinity. However, sometimes the statistic is
undefined, e.g., if a distribution's pdf does not achieve a maximum within
the support of the distribution, the mode is undefined. If the mean is
undefined, then by definition the variance is undefined. E.g. the mean for
Student's T for df = 1 is undefined (no clear way to say it is either + or -
infinity), so the variance = E[(X - mean)**2] is also undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.batch_shape` {#ConditionalTransformedDistribution.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.batch_shape_tensor(name='batch_shape_tensor')` {#ConditionalTransformedDistribution.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.bijector` {#ConditionalTransformedDistribution.bijector}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.cdf(*args, **kwargs)` {#ConditionalTransformedDistribution.cdf}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.copy(**override_parameters_kwargs)` {#ConditionalTransformedDistribution.copy}

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

#### `tf.contrib.distributions.ConditionalTransformedDistribution.covariance(name='covariance')` {#ConditionalTransformedDistribution.covariance}

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

#### `tf.contrib.distributions.ConditionalTransformedDistribution.distribution` {#ConditionalTransformedDistribution.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.dtype` {#ConditionalTransformedDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.entropy(name='entropy')` {#ConditionalTransformedDistribution.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.event_shape` {#ConditionalTransformedDistribution.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.event_shape_tensor(name='event_shape_tensor')` {#ConditionalTransformedDistribution.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.is_continuous` {#ConditionalTransformedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.is_scalar_batch(name='is_scalar_batch')` {#ConditionalTransformedDistribution.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.is_scalar_event(name='is_scalar_event')` {#ConditionalTransformedDistribution.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.log_cdf(*args, **kwargs)` {#ConditionalTransformedDistribution.log_cdf}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.log_prob(*args, **kwargs)` {#ConditionalTransformedDistribution.log_prob}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.log_survival_function(*args, **kwargs)` {#ConditionalTransformedDistribution.log_survival_function}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.mean(name='mean')` {#ConditionalTransformedDistribution.mean}

Mean.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.mode(name='mode')` {#ConditionalTransformedDistribution.mode}

Mode.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.name` {#ConditionalTransformedDistribution.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#ConditionalTransformedDistribution.param_shapes}

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

#### `tf.contrib.distributions.ConditionalTransformedDistribution.param_static_shapes(cls, sample_shape)` {#ConditionalTransformedDistribution.param_static_shapes}

param_shapes with static (i.e. `TensorShape`) shapes.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`. Assumes that the sample's
shape is known statically.

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

#### `tf.contrib.distributions.ConditionalTransformedDistribution.parameters` {#ConditionalTransformedDistribution.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.prob(*args, **kwargs)` {#ConditionalTransformedDistribution.prob}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.reparameterization_type` {#ConditionalTransformedDistribution.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.sample(*args, **kwargs)` {#ConditionalTransformedDistribution.sample}

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.stddev(name='stddev')` {#ConditionalTransformedDistribution.stddev}

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

#### `tf.contrib.distributions.ConditionalTransformedDistribution.survival_function(*args, **kwargs)` {#ConditionalTransformedDistribution.survival_function}

Additional documentation from `ConditionalTransformedDistribution`:

##### `kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.validate_args` {#ConditionalTransformedDistribution.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.ConditionalTransformedDistribution.variance(name='variance')` {#ConditionalTransformedDistribution.variance}

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


