Bernoulli with `probs = nn.sigmoid(logits)`.
- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.__init__(logits=None, dtype=tf.int32, validate_args=False, allow_nan_stats=True, name='BernoulliWithSigmoidProbs')` {#BernoulliWithSigmoidProbs.__init__}




- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.allow_nan_stats` {#BernoulliWithSigmoidProbs.allow_nan_stats}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.batch_shape` {#BernoulliWithSigmoidProbs.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.batch_shape_tensor(name='batch_shape_tensor')` {#BernoulliWithSigmoidProbs.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.cdf(value, name='cdf')` {#BernoulliWithSigmoidProbs.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.copy(**override_parameters_kwargs)` {#BernoulliWithSigmoidProbs.copy}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.covariance(name='covariance')` {#BernoulliWithSigmoidProbs.covariance}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.dtype` {#BernoulliWithSigmoidProbs.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.entropy(name='entropy')` {#BernoulliWithSigmoidProbs.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.event_shape` {#BernoulliWithSigmoidProbs.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.event_shape_tensor(name='event_shape_tensor')` {#BernoulliWithSigmoidProbs.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.is_continuous` {#BernoulliWithSigmoidProbs.is_continuous}




- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.is_scalar_batch(name='is_scalar_batch')` {#BernoulliWithSigmoidProbs.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.is_scalar_event(name='is_scalar_event')` {#BernoulliWithSigmoidProbs.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.log_cdf(value, name='log_cdf')` {#BernoulliWithSigmoidProbs.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.log_prob(value, name='log_prob')` {#BernoulliWithSigmoidProbs.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.log_survival_function(value, name='log_survival_function')` {#BernoulliWithSigmoidProbs.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.logits` {#BernoulliWithSigmoidProbs.logits}

Log-odds of a `1` outcome (vs `0`).


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.mean(name='mean')` {#BernoulliWithSigmoidProbs.mean}

Mean.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.mode(name='mode')` {#BernoulliWithSigmoidProbs.mode}

Mode.

Additional documentation from `Bernoulli`:

Returns `1` if `prob > 0.5` and `0` otherwise.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.name` {#BernoulliWithSigmoidProbs.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#BernoulliWithSigmoidProbs.param_shapes}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.param_static_shapes(cls, sample_shape)` {#BernoulliWithSigmoidProbs.param_static_shapes}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.parameters` {#BernoulliWithSigmoidProbs.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.prob(value, name='prob')` {#BernoulliWithSigmoidProbs.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.probs` {#BernoulliWithSigmoidProbs.probs}

Probability of a `1` outcome (vs `0`).


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.reparameterization_type` {#BernoulliWithSigmoidProbs.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.sample(sample_shape=(), seed=None, name='sample')` {#BernoulliWithSigmoidProbs.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.stddev(name='stddev')` {#BernoulliWithSigmoidProbs.stddev}

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

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.survival_function(value, name='survival_function')` {#BernoulliWithSigmoidProbs.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.validate_args` {#BernoulliWithSigmoidProbs.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.BernoulliWithSigmoidProbs.variance(name='variance')` {#BernoulliWithSigmoidProbs.variance}

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


