Uniform distribution with `a` and `b` parameters.

The PDF of this distribution is constant between [`a`, `b`], and 0 elsewhere.
- - -

#### `tf.contrib.distributions.Uniform.__init__(a=0.0, b=1.0, validate_args=True, allow_nan_stats=False, name='Uniform')` {#Uniform.__init__}

Construct Uniform distributions with `a` and `b`.

The parameters `a` and `b` must be shaped in a way that supports
broadcasting (e.g. `b - a` is a valid operation).

Here are examples without broadcasting:

```python
# Without broadcasting
u1 = Uniform(3.0, 4.0)  # a single uniform distribution [3, 4]
u2 = Uniform([1.0, 2.0], [3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
u3 = Uniform([[1.0, 2.0],
              [3.0, 4.0]],
             [[1.5, 2.5],
              [3.5, 4.5]])  # 4 distributions
```

And with broadcasting:

```python
u1 = Uniform(3.0, [5.0, 6.0, 7.0])  # 3 distributions
```

##### Args:


*  <b>`a`</b>: Floating point tensor, the minimum endpoint.
*  <b>`b`</b>: Floating point tensor, the maximum endpoint. Must be > `a`.
*  <b>`validate_args`</b>: Whether to assert that `a > b`. If `validate_args` is
    `False` and inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.

##### Raises:


*  <b>`InvalidArgumentError`</b>: if `a >= b` and `validate_args=True`.


- - -

#### `tf.contrib.distributions.Uniform.a` {#Uniform.a}




- - -

#### `tf.contrib.distributions.Uniform.allow_nan_stats` {#Uniform.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Uniform.b` {#Uniform.b}




- - -

#### `tf.contrib.distributions.Uniform.batch_shape(name='batch_shape')` {#Uniform.batch_shape}




- - -

#### `tf.contrib.distributions.Uniform.cdf(x, name='cdf')` {#Uniform.cdf}

CDF of observations in `x` under these Uniform distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`. If `x` is `nan`, will
      return `nan`.


- - -

#### `tf.contrib.distributions.Uniform.dtype` {#Uniform.dtype}




- - -

#### `tf.contrib.distributions.Uniform.entropy(name='entropy')` {#Uniform.entropy}

The entropy of Uniform distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Uniform.event_shape(name='event_shape')` {#Uniform.event_shape}




- - -

#### `tf.contrib.distributions.Uniform.from_params(cls, make_safe=True, **kwargs)` {#Uniform.from_params}

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

#### `tf.contrib.distributions.Uniform.get_batch_shape()` {#Uniform.get_batch_shape}




- - -

#### `tf.contrib.distributions.Uniform.get_event_shape()` {#Uniform.get_event_shape}




- - -

#### `tf.contrib.distributions.Uniform.is_continuous` {#Uniform.is_continuous}




- - -

#### `tf.contrib.distributions.Uniform.is_reparameterized` {#Uniform.is_reparameterized}




- - -

#### `tf.contrib.distributions.Uniform.log_cdf(x, name='log_cdf')` {#Uniform.log_cdf}




- - -

#### `tf.contrib.distributions.Uniform.log_pdf(value, name='log_pdf')` {#Uniform.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Uniform.log_pmf(value, name='log_pmf')` {#Uniform.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Uniform.log_prob(x, name='log_prob')` {#Uniform.log_prob}




- - -

#### `tf.contrib.distributions.Uniform.mean(name='mean')` {#Uniform.mean}




- - -

#### `tf.contrib.distributions.Uniform.mode(name='mode')` {#Uniform.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.Uniform.name` {#Uniform.name}




- - -

#### `tf.contrib.distributions.Uniform.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Uniform.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Uniform.param_static_shapes(cls, sample_shape)` {#Uniform.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Uniform.pdf(value, name='pdf')` {#Uniform.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Uniform.pmf(value, name='pmf')` {#Uniform.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Uniform.prob(x, name='prob')` {#Uniform.prob}

The PDF of observations in `x` under these Uniform distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the prob values of `x`. If `x` is `nan`,
      will return `nan`.


- - -

#### `tf.contrib.distributions.Uniform.range(name='range')` {#Uniform.range}

`b - a`.


- - -

#### `tf.contrib.distributions.Uniform.sample(sample_shape=(), seed=None, name='sample')` {#Uniform.sample}

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

#### `tf.contrib.distributions.Uniform.sample_n(n, seed=None, name='sample_n')` {#Uniform.sample_n}

Sample `n` observations from the Uniform Distributions.

##### Args:


*  <b>`n`</b>: `Scalar` `Tensor` of type `int32` or `int64`, the number of
    observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Uniform.std(name='std')` {#Uniform.std}




- - -

#### `tf.contrib.distributions.Uniform.validate_args` {#Uniform.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Uniform.variance(name='variance')` {#Uniform.variance}




