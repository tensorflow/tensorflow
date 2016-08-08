The Laplace distribution with location and scale > 0 parameters.

#### Mathematical details

The PDF of this distribution is:

```f(x | mu, b, b > 0) = 0.5 / b exp(-|x - mu| / b)```

Note that the Laplace distribution can be thought of two exponential
distributions spliced together "back-to-back."
- - -

#### `tf.contrib.distributions.Laplace.__init__(loc, scale, validate_args=True, allow_nan_stats=False, name='Laplace')` {#Laplace.__init__}

Construct Laplace distribution with parameters `loc` and `scale`.

The parameters `loc` and `scale` must be shaped in a way that supports
broadcasting (e.g., `loc / scale` is a valid operation).

##### Args:


*  <b>`loc`</b>: Floating point tensor which characterizes the location (center)
    of the distribution.
*  <b>`scale`</b>: Positive floating point tensor which characterizes the spread of
    the distribution.
*  <b>`validate_args`</b>: Whether to validate input with asserts.  If `validate_args`
    is `False`, and the inputs are invalid, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if `loc` and `scale` are of different dtype.


- - -

#### `tf.contrib.distributions.Laplace.allow_nan_stats` {#Laplace.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Laplace.batch_shape(name='batch_shape')` {#Laplace.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Laplace.cdf(x, name='cdf')` {#Laplace.cdf}

CDF of observations in `x` under the Laplace distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Laplace.dtype` {#Laplace.dtype}




- - -

#### `tf.contrib.distributions.Laplace.entropy(name='entropy')` {#Laplace.entropy}

The entropy of Laplace distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Laplace.event_shape(name='event_shape')` {#Laplace.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Laplace.get_batch_shape()` {#Laplace.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Laplace.get_event_shape()` {#Laplace.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Laplace.is_continuous` {#Laplace.is_continuous}




- - -

#### `tf.contrib.distributions.Laplace.is_reparameterized` {#Laplace.is_reparameterized}




- - -

#### `tf.contrib.distributions.Laplace.loc` {#Laplace.loc}

Distribution parameter for the location.


- - -

#### `tf.contrib.distributions.Laplace.log_cdf(x, name='log_cdf')` {#Laplace.log_cdf}

Log CDF of observations `x` under the Laplace distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Laplace.log_pdf(value, name='log_pdf')` {#Laplace.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Laplace.log_pmf(value, name='log_pmf')` {#Laplace.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Laplace.log_prob(x, name='log_prob')` {#Laplace.log_prob}

Log prob of observations in `x` under these Laplace distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-probability of `x`.


- - -

#### `tf.contrib.distributions.Laplace.mean(name='mean')` {#Laplace.mean}

Mean of this distribution.


- - -

#### `tf.contrib.distributions.Laplace.median(name='median')` {#Laplace.median}

Median of this distribution.


- - -

#### `tf.contrib.distributions.Laplace.mode(name='mode')` {#Laplace.mode}

Mode of this distribution.


- - -

#### `tf.contrib.distributions.Laplace.name` {#Laplace.name}




- - -

#### `tf.contrib.distributions.Laplace.pdf(value, name='pdf')` {#Laplace.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Laplace.pmf(value, name='pmf')` {#Laplace.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Laplace.prob(x, name='pdf')` {#Laplace.prob}

The prob of observations in `x` under the Laplace distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: tensor of dtype `dtype`, the pdf values of `x`.


- - -

#### `tf.contrib.distributions.Laplace.sample(sample_shape=(), seed=None, name='sample')` {#Laplace.sample}

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

#### `tf.contrib.distributions.Laplace.sample_n(n, seed=None, name='sample_n')` {#Laplace.sample_n}

Sample `n` observations from the Laplace Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the parameters.


- - -

#### `tf.contrib.distributions.Laplace.scale` {#Laplace.scale}

Distribution parameter for scale.


- - -

#### `tf.contrib.distributions.Laplace.std(name='std')` {#Laplace.std}

Standard deviation of this distribution.


- - -

#### `tf.contrib.distributions.Laplace.validate_args` {#Laplace.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Laplace.variance(name='variance')` {#Laplace.variance}

Variance of this distribution.


