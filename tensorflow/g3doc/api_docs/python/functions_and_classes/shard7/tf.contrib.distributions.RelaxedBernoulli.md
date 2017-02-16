RelaxedBernoulli distribution with temperature and logits parameters.

The RelaxedBernoulli is a distribution over the unit interval (0,1), which
continuously approximates a Bernoulli. The degree of approximation is
controlled by a temperature: as the temperaturegoes to 0 the RelaxedBernoulli
becomes discrete with a distribution described by the `logits` or `probs`
parameters, as the temperature goes to infinity the RelaxedBernoulli
becomes the constant distribution that is identically 0.5.

The RelaxedBernoulli distribution is a reparameterized continuous
distribution that is the binary special case of the RelaxedOneHotCategorical
distribution (Maddison et al., 2016; Jang et al., 2016). For details on the
binary special case see the appendix of Maddison et al. (2016) where it is
referred to as BinConcrete. If you use this distribution, please cite both
papers.

Some care needs to be taken for loss functions that depend on the
log-probability of RelaxedBernoullis, because computing log-probabilities of
the RelaxedBernoulli can suffer from underflow issues. In many case loss
functions such as these are invariant under invertible transformations of
the random variables. The KL divergence, found in the variational autoencoder
loss, is an example. Because RelaxedBernoullis are sampled by by a Logistic
random variable followed by a `tf.sigmoid` op, one solution is to treat
the Logistic as the random variable and `tf.sigmoid` as downstream. The
KL divergences of two Logistics, which are always followed by a `tf.sigmoid`
op, is equivalent to evaluating KL divergences of RelaxedBernoulli samples.
See Maddison et al., 2016 for more details where this distribution is called
the BinConcrete.

An alternative approach is to evaluate Bernoulli log probability or KL
directly on relaxed samples, as done in Jang et al., 2016. In this case,
guarantees on the loss are usually violated. For instance, using a Bernoulli
KL in a relaxed ELBO is no longer a lower bound on the log marginal
probability of the observation. Thus care and early stopping are important.

#### Examples

Creates three continuous distributions, which approximate 3 Bernoullis with
probabilities (0.1, 0.5, 0.4). Samples from these distributions will be in
the unit interval (0,1).

```python
temperature = 0.5
p = [0.1, 0.5, 0.4]
dist = RelaxedBernoulli(temperature, probs=p)
```

Creates three continuous distributions, which approximate 3 Bernoullis with
logits (-2, 2, 0). Samples from these distributions will be in
the unit interval (0,1).

```python
temperature = 0.5
logits = [-2, 2, 0]
dist = RelaxedBernoulli(temperature, logits=logits)
```

Creates three continuous distributions, whose sigmoid approximate 3 Bernoullis
with logits (-2, 2, 0).

```python
temperature = 0.5
logits = [-2, 2, 0]
dist = Logistic(logits/temperature, 1./temperature)
samples = dist.sample()
sigmoid_samples = tf.sigmoid(samples)
# sigmoid_samples has the same distribution as samples from
# RelaxedBernoulli(temperature, logits=logits)
```

Creates three continuous distributions, which approximate 3 Bernoullis with
logits (-2, 2, 0). Samples from these distributions will be in
the unit interval (0,1). Because the temperature is very low, samples from
these distributions are almost discrete, usually taking values very close to 0
or 1.

```python
temperature = 1e-5
logits = [-2, 2, 0]
dist = RelaxedBernoulli(temperature, logits=logits)
```

Creates three continuous distributions, which approximate 3 Bernoullis with
logits (-2, 2, 0). Samples from these distributions will be in
the unit interval (0,1). Because the temperature is very high, samples from
these distributions are usually close to the (0.5, 0.5, 0.5) vector.

```python
temperature = 100
logits = [-2, 2, 0]
dist = RelaxedBernoulli(temperature, logits=logits)
```

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
A Continuous Relaxation of Discrete Random Variables. 2016.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with
Gumbel-Softmax. 2016.
- - -

#### `tf.contrib.distributions.RelaxedBernoulli.__init__(temperature, logits=None, probs=None, validate_args=False, allow_nan_stats=True, name='RelaxedBernoulli')` {#RelaxedBernoulli.__init__}

Construct RelaxedBernoulli distributions.

##### Args:


*  <b>`temperature`</b>: An 0-D `Tensor`, representing the temperature
    of a set of RelaxedBernoulli distributions. The temperature should be
    positive.
*  <b>`logits`</b>: An N-D `Tensor` representing the log-odds
    of a positive event. Each entry in the `Tensor` parametrizes
    an independent RelaxedBernoulli distribution where the probability of an
    event is sigmoid(logits). Only one of `logits` or `probs` should be
    passed in.
*  <b>`probs`</b>: An N-D `Tensor` representing the probability of a positive event.
    Each entry in the `Tensor` parameterizes an independent Bernoulli
    distribution. Only one of `logits` or `probs` should be passed in.
*  <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
*  <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
*  <b>`name`</b>: Python `str` name prefixed to Ops created by this class.

##### Raises:


*  <b>`ValueError`</b>: If both `probs` and `logits` are passed, or if neither.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.allow_nan_stats` {#RelaxedBernoulli.allow_nan_stats}

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

#### `tf.contrib.distributions.RelaxedBernoulli.batch_shape` {#RelaxedBernoulli.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.batch_shape_tensor(name='batch_shape_tensor')` {#RelaxedBernoulli.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.bijector` {#RelaxedBernoulli.bijector}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.cdf(value, name='cdf')` {#RelaxedBernoulli.cdf}

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

#### `tf.contrib.distributions.RelaxedBernoulli.copy(**override_parameters_kwargs)` {#RelaxedBernoulli.copy}

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

#### `tf.contrib.distributions.RelaxedBernoulli.covariance(name='covariance')` {#RelaxedBernoulli.covariance}

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

#### `tf.contrib.distributions.RelaxedBernoulli.distribution` {#RelaxedBernoulli.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.dtype` {#RelaxedBernoulli.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.entropy(name='entropy')` {#RelaxedBernoulli.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.event_shape` {#RelaxedBernoulli.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.event_shape_tensor(name='event_shape_tensor')` {#RelaxedBernoulli.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.is_continuous` {#RelaxedBernoulli.is_continuous}




- - -

#### `tf.contrib.distributions.RelaxedBernoulli.is_scalar_batch(name='is_scalar_batch')` {#RelaxedBernoulli.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.is_scalar_event(name='is_scalar_event')` {#RelaxedBernoulli.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.log_cdf(value, name='log_cdf')` {#RelaxedBernoulli.log_cdf}

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

#### `tf.contrib.distributions.RelaxedBernoulli.log_prob(value, name='log_prob')` {#RelaxedBernoulli.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.log_survival_function(value, name='log_survival_function')` {#RelaxedBernoulli.log_survival_function}

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

#### `tf.contrib.distributions.RelaxedBernoulli.logits` {#RelaxedBernoulli.logits}

Log-odds of `1`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.mean(name='mean')` {#RelaxedBernoulli.mean}

Mean.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.mode(name='mode')` {#RelaxedBernoulli.mode}

Mode.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.name` {#RelaxedBernoulli.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#RelaxedBernoulli.param_shapes}

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

#### `tf.contrib.distributions.RelaxedBernoulli.param_static_shapes(cls, sample_shape)` {#RelaxedBernoulli.param_static_shapes}

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

#### `tf.contrib.distributions.RelaxedBernoulli.parameters` {#RelaxedBernoulli.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.prob(value, name='prob')` {#RelaxedBernoulli.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.probs` {#RelaxedBernoulli.probs}

Probability of `1`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.reparameterization_type` {#RelaxedBernoulli.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.sample(sample_shape=(), seed=None, name='sample')` {#RelaxedBernoulli.sample}

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

#### `tf.contrib.distributions.RelaxedBernoulli.stddev(name='stddev')` {#RelaxedBernoulli.stddev}

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

#### `tf.contrib.distributions.RelaxedBernoulli.survival_function(value, name='survival_function')` {#RelaxedBernoulli.survival_function}

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

#### `tf.contrib.distributions.RelaxedBernoulli.temperature` {#RelaxedBernoulli.temperature}

Distribution parameter for the location.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.validate_args` {#RelaxedBernoulli.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.RelaxedBernoulli.variance(name='variance')` {#RelaxedBernoulli.variance}

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


