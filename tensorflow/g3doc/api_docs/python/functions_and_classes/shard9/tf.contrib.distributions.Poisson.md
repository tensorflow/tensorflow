Poisson distribution.

The Poisson distribution is parameterized by `lam`, the rate parameter.

The pmf of this distribution is:

```

pmf(k) = e^(-lam) * lam^k / k!,  k >= 0
```
- - -

#### `tf.contrib.distributions.Poisson.__init__(lam, validate_args=True, allow_nan_stats=False, name='Poisson')` {#Poisson.__init__}

Construct Poisson distributions.

##### Args:


*  <b>`lam`</b>: Floating point tensor, the rate parameter of the
    distribution(s). `lam` must be positive.
*  <b>`validate_args`</b>: Whether to assert that `lam > 0` as well as inputs to
    pmf computations are non-negative integers. If validate_args is
    `False`, then `pmf` computations might return NaN, as well as
    can be evaluated at any real value.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: A name for this distribution.


- - -

#### `tf.contrib.distributions.Poisson.allow_nan_stats` {#Poisson.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Poisson.batch_shape(name='batch_shape')` {#Poisson.batch_shape}




- - -

#### `tf.contrib.distributions.Poisson.cdf(x, name='cdf')` {#Poisson.cdf}

Cumulative density function.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor with dtype `dtype` and whose shape
    can be broadcast with `self.lam`.
*  <b>`name`</b>: A name for this operation.

##### Returns:

  The CDF of the events.


- - -

#### `tf.contrib.distributions.Poisson.dtype` {#Poisson.dtype}




- - -

#### `tf.contrib.distributions.Poisson.entropy(name='entropy')` {#Poisson.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.Poisson.event_shape(name='event_shape')` {#Poisson.event_shape}




- - -

#### `tf.contrib.distributions.Poisson.from_params(cls, make_safe=True, **kwargs)` {#Poisson.from_params}

Given (unconstrained) parameters, return an instantiated distribution.

Subclasses should implement a static method `_safe_transforms` that returns
a dict of parameter transforms, which will be used if `make_safe = True`.

Example usage:

```
# Let's say we want a sample of size (batch_size, 10)
shapes = MultiVariateNormalDiag.param_shapes([batch_size, 10])

# shapes has a Tensor shape for mu and sigma
# shapes == {
#   'mu': tf.constant([batch_size, 10]),
#   'sigma': tf.constant([batch_size, 10]),
# }

# Here we parameterize mu and sigma with the output of a linear
# layer. Note that sigma is unconstrained.
params = {}
for name, shape in shapes.items():
  params[name] = linear(x, shape[1])

# Note that you can forward other kwargs to the `Distribution`, like
# `allow_nan_stats` or `name`.
mvn = MultiVariateNormalDiag.from_params(**params, allow_nan_stats=True)
```

Distribution parameters may have constraints (e.g. `sigma` must be positive
for a `Normal` distribution) and the `from_params` method will apply default
parameter transforms. If a user wants to use their own transform, they can
apply it externally and set `make_safe=False`.

##### Args:


*  <b>`make_safe`</b>: Whether the `params` should be constrained. If True,
    `from_params` will apply default parameter transforms. If False, no
    parameter transforms will be applied.
*  <b>`**kwargs`</b>: dict of parameters for the distribution.

##### Returns:

  A distribution parameterized by possibly transformed parameters in
  `kwargs`.

##### Raises:


*  <b>`TypeError`</b>: if `make_safe` is `True` but `_safe_transforms` is not
    implemented directly for `cls`.


- - -

#### `tf.contrib.distributions.Poisson.get_batch_shape()` {#Poisson.get_batch_shape}




- - -

#### `tf.contrib.distributions.Poisson.get_event_shape()` {#Poisson.get_event_shape}




- - -

#### `tf.contrib.distributions.Poisson.is_continuous` {#Poisson.is_continuous}




- - -

#### `tf.contrib.distributions.Poisson.is_reparameterized` {#Poisson.is_reparameterized}




- - -

#### `tf.contrib.distributions.Poisson.lam` {#Poisson.lam}

Rate parameter.


- - -

#### `tf.contrib.distributions.Poisson.log_cdf(x, name='log_cdf')` {#Poisson.log_cdf}

Log cumulative density function.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor with dtype `dtype` and whose shape
    can be broadcast with `self.lam`.
*  <b>`name`</b>: A name for this operation.

##### Returns:

  The Log CDF of the events.


- - -

#### `tf.contrib.distributions.Poisson.log_pdf(value, name='log_pdf')` {#Poisson.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Poisson.log_pmf(value, name='log_pmf')` {#Poisson.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Poisson.log_prob(x, name='log_prob')` {#Poisson.log_prob}

Log probability mass function.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor with dtype `dtype` and whose shape
    can be broadcast with `self.lam`. `x` is only legal if it is
    non-negative and its components are equal to integer values.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  The log-probabilities of the events.


- - -

#### `tf.contrib.distributions.Poisson.mean(name='mean')` {#Poisson.mean}

Mean of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`mean`</b>: `Tensor` of the same type and shape as `lam`.


- - -

#### `tf.contrib.distributions.Poisson.mode(name='mode')` {#Poisson.mode}

Mode of the distribution.

Note that when `lam` is an integer, there are actually two modes.
Namely, `lam` and `lam - 1` are both modes. Here we return
only the larger of the two modes.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`mode`</b>: `Tensor` of the same type and shape as `lam`.


- - -

#### `tf.contrib.distributions.Poisson.name` {#Poisson.name}




- - -

#### `tf.contrib.distributions.Poisson.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Poisson.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Poisson.param_static_shapes(cls, sample_shape)` {#Poisson.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Poisson.pdf(value, name='pdf')` {#Poisson.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Poisson.pmf(value, name='pmf')` {#Poisson.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Poisson.prob(x, name='prob')` {#Poisson.prob}

Probability mass function.

##### Args:


*  <b>`x`</b>: Non-negative floating point tensor with dtype `dtype` and whose shape
    can be broadcast with `self.lam`. `x` is only legal if it is
    non-negative and its components are equal to integer values.
*  <b>`name`</b>: A name for this operation.

##### Returns:

  The probabilities of the events.


- - -

#### `tf.contrib.distributions.Poisson.sample(sample_shape=(), seed=None, name='sample')` {#Poisson.sample}

Generate samples of the specified shape for each batched distribution.

Note that a call to `sample()` without arguments will generate a single
sample per batched distribution.

##### Args:


*  <b>`sample_shape`</b>: Rank 1 `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of dtype `self.dtype` and shape
      `sample_shape + self.batch_shape + self.event_shape`.


- - -

#### `tf.contrib.distributions.Poisson.sample_n(n, seed=None, name='sample_n')` {#Poisson.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Poisson.std(name='std')` {#Poisson.std}

Standard deviation of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`std`</b>: `Tensor` of the same type and shape as `lam`.


- - -

#### `tf.contrib.distributions.Poisson.validate_args` {#Poisson.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Poisson.variance(name='variance')` {#Poisson.variance}

Variance of the distribution.

##### Args:


*  <b>`name`</b>: Name for the op.

##### Returns:


*  <b>`variance`</b>: `Tensor` of the same type and shape as `lam`.


