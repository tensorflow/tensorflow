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


*  <b>`sample_shape`</b>: `int32` `Tensor` or tuple or list. Shape of the generated
    samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of dtype `self.dtype` and shape
      `sample_shape + self.batch_shape + self.event_shape`.


- - -

#### `tf.contrib.distributions.Uniform.sample_n(n, seed=None, name='sample_n')` {#Uniform.sample_n}

Sample `n` observations from the Uniform Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
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




