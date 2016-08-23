Categorical distribution.

The categorical distribution is parameterized by the log-probabilities
of a set of classes.
- - -

#### `tf.contrib.distributions.Categorical.__init__(logits, dtype=tf.int32, validate_args=True, allow_nan_stats=False, name='Categorical')` {#Categorical.__init__}

Initialize Categorical distributions using class log-probabilities.

##### Args:


*  <b>`logits`</b>: An N-D `Tensor`, `N >= 1`, representing the log probabilities
      of a set of Categorical distributions. The first `N - 1` dimensions
      index into a batch of independent distributions and the last dimension
      indexes into the classes.
*  <b>`dtype`</b>: The type of the event samples (default: int32).
*  <b>`validate_args`</b>: Unused in this distribution.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: A name for this distribution (optional).


- - -

#### `tf.contrib.distributions.Categorical.allow_nan_stats` {#Categorical.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Categorical.batch_shape(name='batch_shape')` {#Categorical.batch_shape}




- - -

#### `tf.contrib.distributions.Categorical.cdf(value, name='cdf')` {#Categorical.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Categorical.dtype` {#Categorical.dtype}




- - -

#### `tf.contrib.distributions.Categorical.entropy(name='sample')` {#Categorical.entropy}




- - -

#### `tf.contrib.distributions.Categorical.event_shape(name='event_shape')` {#Categorical.event_shape}




- - -

#### `tf.contrib.distributions.Categorical.from_params(cls, make_safe=True, **kwargs)` {#Categorical.from_params}

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

#### `tf.contrib.distributions.Categorical.get_batch_shape()` {#Categorical.get_batch_shape}




- - -

#### `tf.contrib.distributions.Categorical.get_event_shape()` {#Categorical.get_event_shape}




- - -

#### `tf.contrib.distributions.Categorical.is_continuous` {#Categorical.is_continuous}




- - -

#### `tf.contrib.distributions.Categorical.is_reparameterized` {#Categorical.is_reparameterized}




- - -

#### `tf.contrib.distributions.Categorical.log_cdf(value, name='log_cdf')` {#Categorical.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Categorical.log_pdf(value, name='log_pdf')` {#Categorical.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Categorical.log_pmf(value, name='log_pmf')` {#Categorical.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Categorical.log_prob(k, name='log_prob')` {#Categorical.log_prob}

Log-probability of class `k`.

##### Args:


*  <b>`k`</b>: `int32` or `int64` Tensor. Must be broadcastable with a `batch_shape`
    `Tensor`.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  The log-probabilities of the classes indexed by `k`


- - -

#### `tf.contrib.distributions.Categorical.logits` {#Categorical.logits}




- - -

#### `tf.contrib.distributions.Categorical.mean(name='mean')` {#Categorical.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.Categorical.mode(name='mode')` {#Categorical.mode}




- - -

#### `tf.contrib.distributions.Categorical.name` {#Categorical.name}




- - -

#### `tf.contrib.distributions.Categorical.num_classes` {#Categorical.num_classes}

Scalar `int32` tensor: the number of classes.


- - -

#### `tf.contrib.distributions.Categorical.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Categorical.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Categorical.param_static_shapes(cls, sample_shape)` {#Categorical.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Categorical.pdf(value, name='pdf')` {#Categorical.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Categorical.pmf(value, name='pmf')` {#Categorical.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Categorical.prob(k, name='prob')` {#Categorical.prob}

Probability of class `k`.

##### Args:


*  <b>`k`</b>: `int32` or `int64` Tensor. Must be broadcastable with logits.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  The probabilities of the classes indexed by `k`


- - -

#### `tf.contrib.distributions.Categorical.sample(sample_shape=(), seed=None, name='sample')` {#Categorical.sample}

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

#### `tf.contrib.distributions.Categorical.sample_n(n, seed=None, name='sample_n')` {#Categorical.sample_n}

Sample `n` observations from the Categorical distribution.

##### Args:


*  <b>`n`</b>: 0-D.  Number of independent samples to draw for each distribution.
*  <b>`seed`</b>: Random seed (optional).
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  An `int64` `Tensor` with shape `[n, batch_shape, event_shape]`


- - -

#### `tf.contrib.distributions.Categorical.std(name='std')` {#Categorical.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.Categorical.validate_args` {#Categorical.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Categorical.variance(name='variance')` {#Categorical.variance}

Variance of the distribution.


