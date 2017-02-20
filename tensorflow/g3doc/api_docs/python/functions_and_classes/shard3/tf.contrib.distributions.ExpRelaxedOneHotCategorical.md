ExpRelaxedOneHotCategorical distribution with temperature and logits.

An ExpRelaxedOneHotCategorical distribution is a log-transformed
RelaxedOneHotCategorical distribution. The RelaxedOneHotCategorical is a
distribution over random probability vectors, vectors of positive real
values that sum to one, which continuously approximates a OneHotCategorical.
The degree of approximation is controlled by a temperature: as the temperature
goes to 0 the RelaxedOneHotCategorical becomes discrete with a distribution
described by the logits, as the temperature goes to infinity the
RelaxedOneHotCategorical becomes the constant distribution that is identically
the constant vector of (1/event_size, ..., 1/event_size).

Because computing log-probabilities of the RelaxedOneHotCategorical can
suffer from underflow issues, this class is one solution for loss
functions that depend on log-probabilities, such as the KL Divergence found
in the variational autoencoder loss. The KL divergence between two
distributions is invariant under invertible transformations, so evaluating
KL divergences of ExpRelaxedOneHotCategorical samples, which are always
followed by a `tf.exp` op, is equivalent to evaluating KL divergences of
RelaxedOneHotCategorical samples. See the appendix of Maddison et al., 2016
for more mathematical details, where this distribution is called the
ExpConcrete.

#### Examples

Creates a continuous distribution, whoe exp approximates a 3-class one-hot
categorical distiribution. The 2nd class is the most likely to be the
largest component in samples drawn from this distribution. If those samples
are followed by a `tf.exp` op, then they are distributed as a relaxed onehot
categorical.

```python
temperature = 0.5
p = [0.1, 0.5, 0.4]
dist = ExpRelaxedOneHotCategorical(temperature, probs=p)
samples = dist.sample()
exp_samples = tf.exp(samples)
# exp_samples has the same distribution as samples from
# RelaxedOneHotCategorical(temperature, probs=p)
```

Creates a continuous distribution, whose exp approximates a 3-class one-hot
categorical distiribution. The 2nd class is the most likely to be the
largest component in samples drawn from this distribution.

```python
temperature = 0.5
logits = [-2, 2, 0]
dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
samples = dist.sample()
exp_samples = tf.exp(samples)
# exp_samples has the same distribution as samples from
# RelaxedOneHotCategorical(temperature, probs=p)
```

Creates a continuous distribution, whose exp approximates a 3-class one-hot
categorical distiribution. Because the temperature is very low, samples from
this distribution are almost discrete, with one component almost 0 and the
others very negative. The 2nd class is the most likely to be the largest
component in samples drawn from this distribution.

```python
temperature = 1e-5
logits = [-2, 2, 0]
dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
samples = dist.sample()
exp_samples = tf.exp(samples)
# exp_samples has the same distribution as samples from
# RelaxedOneHotCategorical(temperature, probs=p)
```

Creates a continuous distribution, whose exp approximates a 3-class one-hot
categorical distiribution. Because the temperature is very high, samples from
this distribution are usually close to the (-log(3), -log(3), -log(3)) vector.
The 2nd class is still the most likely to be the largest component
in samples drawn from this distribution.

```python
temperature = 10
logits = [-2, 2, 0]
dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
samples = dist.sample()
exp_samples = tf.exp(samples)
# exp_samples has the same distribution as samples from
# RelaxedOneHotCategorical(temperature, probs=p)
```

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
A Continuous Relaxation of Discrete Random Variables. 2016.
- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.__init__(temperature, logits=None, probs=None, dtype=tf.float32, validate_args=False, allow_nan_stats=True, name='ExpRelaxedOneHotCategorical')` {#ExpRelaxedOneHotCategorical.__init__}

Initialize ExpRelaxedOneHotCategorical using class log-probabilities.

##### Args:


*  <b>`temperature`</b>: An 0-D `Tensor`, representing the temperature
    of a set of ExpRelaxedCategorical distributions. The temperature should
    be positive.
*  <b>`logits`</b>: An N-D `Tensor`, `N >= 1`, representing the log probabilities
    of a set of ExpRelaxedCategorical distributions. The first
    `N - 1` dimensions index into a batch of independent distributions and
    the last dimension represents a vector of logits for each class. Only
    one of `logits` or `probs` should be passed in.
*  <b>`probs`</b>: An N-D `Tensor`, `N >= 1`, representing the probabilities
    of a set of ExpRelaxedCategorical distributions. The first
    `N - 1` dimensions index into a batch of independent distributions and
    the last dimension represents a vector of probabilities for each
    class. Only one of `logits` or `probs` should be passed in.
*  <b>`dtype`</b>: The type of the event samples (default: int32).
*  <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
*  <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
*  <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.allow_nan_stats` {#ExpRelaxedOneHotCategorical.allow_nan_stats}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.batch_shape` {#ExpRelaxedOneHotCategorical.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.batch_shape_tensor(name='batch_shape_tensor')` {#ExpRelaxedOneHotCategorical.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.cdf(value, name='cdf')` {#ExpRelaxedOneHotCategorical.cdf}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.copy(**override_parameters_kwargs)` {#ExpRelaxedOneHotCategorical.copy}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.covariance(name='covariance')` {#ExpRelaxedOneHotCategorical.covariance}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.dtype` {#ExpRelaxedOneHotCategorical.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.entropy(name='entropy')` {#ExpRelaxedOneHotCategorical.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.event_shape` {#ExpRelaxedOneHotCategorical.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.event_shape_tensor(name='event_shape_tensor')` {#ExpRelaxedOneHotCategorical.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.event_size` {#ExpRelaxedOneHotCategorical.event_size}

Scalar `int32` tensor: the number of classes.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.is_continuous` {#ExpRelaxedOneHotCategorical.is_continuous}




- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.is_scalar_batch(name='is_scalar_batch')` {#ExpRelaxedOneHotCategorical.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.is_scalar_event(name='is_scalar_event')` {#ExpRelaxedOneHotCategorical.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.log_cdf(value, name='log_cdf')` {#ExpRelaxedOneHotCategorical.log_cdf}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.log_prob(value, name='log_prob')` {#ExpRelaxedOneHotCategorical.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.log_survival_function(value, name='log_survival_function')` {#ExpRelaxedOneHotCategorical.log_survival_function}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.logits` {#ExpRelaxedOneHotCategorical.logits}

Vector of coordinatewise logits.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.mean(name='mean')` {#ExpRelaxedOneHotCategorical.mean}

Mean.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.mode(name='mode')` {#ExpRelaxedOneHotCategorical.mode}

Mode.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.name` {#ExpRelaxedOneHotCategorical.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#ExpRelaxedOneHotCategorical.param_shapes}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.param_static_shapes(cls, sample_shape)` {#ExpRelaxedOneHotCategorical.param_static_shapes}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.parameters` {#ExpRelaxedOneHotCategorical.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.prob(value, name='prob')` {#ExpRelaxedOneHotCategorical.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.probs` {#ExpRelaxedOneHotCategorical.probs}

Vector of probabilities summing to one.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.reparameterization_type` {#ExpRelaxedOneHotCategorical.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.sample(sample_shape=(), seed=None, name='sample')` {#ExpRelaxedOneHotCategorical.sample}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.stddev(name='stddev')` {#ExpRelaxedOneHotCategorical.stddev}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.survival_function(value, name='survival_function')` {#ExpRelaxedOneHotCategorical.survival_function}

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

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.temperature` {#ExpRelaxedOneHotCategorical.temperature}

Batchwise temperature tensor of a RelaxedCategorical.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.validate_args` {#ExpRelaxedOneHotCategorical.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.ExpRelaxedOneHotCategorical.variance(name='variance')` {#ExpRelaxedOneHotCategorical.variance}

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


