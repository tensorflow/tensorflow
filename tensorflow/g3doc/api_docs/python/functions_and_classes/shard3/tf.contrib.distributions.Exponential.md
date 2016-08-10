The Exponential distribution with rate parameter lam.

The PDF of this distribution is:

```prob(x) = (lam * e^(-lam * x)), x > 0```

Note that the Exponential distribution is a special case of the Gamma
distribution, with Exponential(lam) = Gamma(1, lam).
- - -

#### `tf.contrib.distributions.Exponential.__init__(lam, validate_args=True, allow_nan_stats=False, name='Exponential')` {#Exponential.__init__}

Construct Exponential distribution with parameter `lam`.

##### Args:


*  <b>`lam`</b>: Floating point tensor, the rate of the distribution(s).
    `lam` must contain only positive values.
*  <b>`validate_args`</b>: Whether to assert that `lam > 0`, and that `x > 0` in the
    methods `prob(x)` and `log_prob(x)`.  If `validate_args` is `False`
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member. If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prepend to all ops created by this distribution.


- - -

#### `tf.contrib.distributions.Exponential.allow_nan_stats` {#Exponential.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Exponential.alpha` {#Exponential.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.Exponential.batch_shape(name='batch_shape')` {#Exponential.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Exponential.beta` {#Exponential.beta}

Inverse scale parameter.


- - -

#### `tf.contrib.distributions.Exponential.cdf(x, name='cdf')` {#Exponential.cdf}

CDF of observations `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Exponential.dtype` {#Exponential.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Exponential.entropy(name='entropy')` {#Exponential.entropy}

The entropy of Gamma distribution(s).

This is defined to be

```
entropy = alpha - log(beta) + log(Gamma(alpha))
             + (1-alpha)digamma(alpha)
```

where digamma(alpha) is the digamma function.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Exponential.event_shape(name='event_shape')` {#Exponential.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Exponential.get_batch_shape()` {#Exponential.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.Exponential.get_event_shape()` {#Exponential.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.Exponential.is_continuous` {#Exponential.is_continuous}




- - -

#### `tf.contrib.distributions.Exponential.is_reparameterized` {#Exponential.is_reparameterized}




- - -

#### `tf.contrib.distributions.Exponential.lam` {#Exponential.lam}




- - -

#### `tf.contrib.distributions.Exponential.log_cdf(x, name='log_cdf')` {#Exponential.log_cdf}

Log CDF of observations `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Exponential.log_pdf(value, name='log_pdf')` {#Exponential.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Exponential.log_pmf(value, name='log_pmf')` {#Exponential.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Exponential.log_prob(x, name='log_prob')` {#Exponential.log_prob}

Log prob of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Exponential.mean(name='mean')` {#Exponential.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.Exponential.mode(name='mode')` {#Exponential.mode}

Mode of each batch member.

The mode of a gamma distribution is `(alpha - 1) / beta` when `alpha > 1`,
and `NaN` otherwise.  If `self.allow_nan_stats` is `False`, an exception
will be raised rather than returning `NaN`.

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The mode for every batch member, a `Tensor` with same `dtype` as self.


- - -

#### `tf.contrib.distributions.Exponential.name` {#Exponential.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Exponential.pdf(value, name='pdf')` {#Exponential.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Exponential.pmf(value, name='pmf')` {#Exponential.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Exponential.prob(x, name='prob')` {#Exponential.prob}

Pdf of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the PDFs of `x`

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Exponential.sample(sample_shape=(), seed=None, name='sample')` {#Exponential.sample}

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

#### `tf.contrib.distributions.Exponential.sample_n(n, seed=None, name='sample_n')` {#Exponential.sample_n}

Sample `n` observations from the Exponential Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by the hyperparameters.


- - -

#### `tf.contrib.distributions.Exponential.std(name='std')` {#Exponential.std}

Standard deviation of this distribution.


- - -

#### `tf.contrib.distributions.Exponential.validate_args` {#Exponential.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Exponential.variance(name='variance')` {#Exponential.variance}

Variance of each batch member.


