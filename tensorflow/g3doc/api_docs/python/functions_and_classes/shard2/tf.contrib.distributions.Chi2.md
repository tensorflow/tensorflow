The Chi2 distribution with degrees of freedom df.

The PDF of this distribution is:

```pdf(x) = (x^(df/2 - 1)e^(-x/2))/(2^(df/2)Gamma(df/2)), x > 0```

Note that the Chi2 distribution is a special case of the Gamma distribution,
with Chi2(df) = Gamma(df/2, 1/2).
- - -

#### `tf.contrib.distributions.Chi2.__init__(df, validate_args=True, allow_nan_stats=False, name='Chi2')` {#Chi2.__init__}

Construct Chi2 distributions with parameter `df`.

##### Args:


*  <b>`df`</b>: Floating point tensor, the degrees of freedom of the
    distribution(s).  `df` must contain only positive values.
*  <b>`validate_args`</b>: Whether to assert that `df > 0`, and that `x > 0` in the
    methods `prob(x)` and `log_prob(x)`. If `validate_args` is `False`
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prepend to all ops created by this distribution.


- - -

#### `tf.contrib.distributions.Chi2.allow_nan_stats` {#Chi2.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Chi2.alpha` {#Chi2.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.Chi2.batch_shape(name='batch_shape')` {#Chi2.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Chi2.beta` {#Chi2.beta}

Inverse scale parameter.


- - -

#### `tf.contrib.distributions.Chi2.cdf(x, name='cdf')` {#Chi2.cdf}

CDF of observations `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Chi2.df` {#Chi2.df}




- - -

#### `tf.contrib.distributions.Chi2.dtype` {#Chi2.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Chi2.entropy(name='entropy')` {#Chi2.entropy}

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

#### `tf.contrib.distributions.Chi2.event_shape(name='event_shape')` {#Chi2.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Chi2.get_batch_shape()` {#Chi2.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.Chi2.get_event_shape()` {#Chi2.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.Chi2.is_continuous` {#Chi2.is_continuous}




- - -

#### `tf.contrib.distributions.Chi2.is_reparameterized` {#Chi2.is_reparameterized}




- - -

#### `tf.contrib.distributions.Chi2.log_cdf(x, name='log_cdf')` {#Chi2.log_cdf}

Log CDF of observations `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Chi2.log_pdf(value, name='log_pdf')` {#Chi2.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Chi2.log_pmf(value, name='log_pmf')` {#Chi2.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Chi2.log_prob(x, name='log_prob')` {#Chi2.log_prob}

Log prob of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Chi2.mean(name='mean')` {#Chi2.mean}

Mean of each batch member.


- - -

#### `tf.contrib.distributions.Chi2.mode(name='mode')` {#Chi2.mode}

Mode of each batch member.

The mode of a gamma distribution is `(alpha - 1) / beta` when `alpha > 1`,
and `NaN` otherwise.  If `self.allow_nan_stats` is `False`, an exception
will be raised rather than returning `NaN`.

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The mode for every batch member, a `Tensor` with same `dtype` as self.


- - -

#### `tf.contrib.distributions.Chi2.name` {#Chi2.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Chi2.pdf(value, name='pdf')` {#Chi2.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Chi2.pmf(value, name='pmf')` {#Chi2.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Chi2.prob(x, name='prob')` {#Chi2.prob}

Pdf of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the PDFs of `x`

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Chi2.sample(sample_shape=(), seed=None, name='sample')` {#Chi2.sample}

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

#### `tf.contrib.distributions.Chi2.sample_n(n, seed=None, name='sample_n')` {#Chi2.sample_n}

Draws `n` samples from the Gamma distribution(s).

See the doc for tf.random_gamma for further detail.

##### Args:


*  <b>`n`</b>: Python integer, the number of observations to sample from each
    distribution.
*  <b>`seed`</b>: Python integer, the random seed for this operation.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Chi2.std(name='std')` {#Chi2.std}

Standard deviation of this distribution.


- - -

#### `tf.contrib.distributions.Chi2.validate_args` {#Chi2.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Chi2.variance(name='variance')` {#Chi2.variance}

Variance of each batch member.


