The `InverseGamma` distribution with parameter alpha and beta.

The parameters are the shape and inverse scale parameters alpha, beta.

The PDF of this distribution is:

```pdf(x) = (beta^alpha)/Gamma(alpha)(x^(-alpha-1))e^(-beta/x), x > 0```

and the CDF of this distribution is:

```cdf(x) =  GammaInc(alpha, beta / x) / Gamma(alpha), x > 0```

where GammaInc is the upper incomplete Gamma function.

Examples:

```python
dist = InverseGamma(alpha=3.0, beta=2.0)
dist2 = InverseGamma(alpha=[3.0, 4.0], beta=[2.0, 3.0])
```
- - -

#### `tf.contrib.distributions.InverseGamma.__init__(alpha, beta, validate_args=True, allow_nan_stats=False, name='InverseGamma')` {#InverseGamma.__init__}

Construct InverseGamma distributions with parameters `alpha` and `beta`.

The parameters `alpha` and `beta` must be shaped in a way that supports
broadcasting (e.g. `alpha + beta` is a valid operation).

##### Args:


*  <b>`alpha`</b>: Floating point tensor, the shape params of the
    distribution(s).
    alpha must contain only positive values.
*  <b>`beta`</b>: Floating point tensor, the scale params of the distribution(s).
    beta must contain only positive values.
*  <b>`validate_args`</b>: Whether to assert that `a > 0, b > 0`, and that `x > 0` in
    the methods `prob(x)` and `log_prob(x)`.  If `validate_args` is `False`
    and the inputs are invalid, correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prepend to all ops created by this distribution.

##### Raises:


*  <b>`TypeError`</b>: if `alpha` and `beta` are different dtypes.


- - -

#### `tf.contrib.distributions.InverseGamma.allow_nan_stats` {#InverseGamma.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.InverseGamma.alpha` {#InverseGamma.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.InverseGamma.batch_shape(name='batch_shape')` {#InverseGamma.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.InverseGamma.beta` {#InverseGamma.beta}

Scale parameter.


- - -

#### `tf.contrib.distributions.InverseGamma.cdf(x, name='cdf')` {#InverseGamma.cdf}

CDF of observations `x` under these InverseGamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.InverseGamma.dtype` {#InverseGamma.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.InverseGamma.entropy(name='entropy')` {#InverseGamma.entropy}

The entropy of these InverseGamma distribution(s).

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

#### `tf.contrib.distributions.InverseGamma.event_shape(name='event_shape')` {#InverseGamma.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.InverseGamma.get_batch_shape()` {#InverseGamma.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.InverseGamma.get_event_shape()` {#InverseGamma.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  `TensorShape` object.


- - -

#### `tf.contrib.distributions.InverseGamma.is_continuous` {#InverseGamma.is_continuous}




- - -

#### `tf.contrib.distributions.InverseGamma.is_reparameterized` {#InverseGamma.is_reparameterized}




- - -

#### `tf.contrib.distributions.InverseGamma.log_cdf(x, name='log_cdf')` {#InverseGamma.log_cdf}

Log CDF of observations `x` under these InverseGamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.InverseGamma.log_pdf(value, name='log_pdf')` {#InverseGamma.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.InverseGamma.log_pmf(value, name='log_pmf')` {#InverseGamma.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.InverseGamma.log_prob(x, name='log_prob')` {#InverseGamma.log_prob}

Log prob of observations in `x` under these InverseGamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.InverseGamma.mean(name='mean')` {#InverseGamma.mean}

Mean of each batch member.

The mean of an inverse gamma distribution is `beta / (alpha - 1)`,
when `alpha > 1`, and `NaN` otherwise.  If `self.allow_nan_stats` is
`False`, an exception will be raised rather than returning `NaN`

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The mean for every batch member, a `Tensor` with same `dtype` as self.


- - -

#### `tf.contrib.distributions.InverseGamma.mode(name='mode')` {#InverseGamma.mode}

Mode of each batch member.

The mode of an inverse gamma distribution is `beta / (alpha + 1)`.

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The mode for every batch member, a `Tensor` with same `dtype` as self.


- - -

#### `tf.contrib.distributions.InverseGamma.name` {#InverseGamma.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.InverseGamma.pdf(value, name='pdf')` {#InverseGamma.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.InverseGamma.pmf(value, name='pmf')` {#InverseGamma.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.InverseGamma.prob(x, name='prob')` {#InverseGamma.prob}

Pdf of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: tensor of dtype `dtype`, the PDFs of `x`

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.InverseGamma.sample(sample_shape=(), seed=None, name='sample')` {#InverseGamma.sample}

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

#### `tf.contrib.distributions.InverseGamma.sample_n(n, seed=None, name='sample_n')` {#InverseGamma.sample_n}

Draws `n` samples from these InverseGamma distribution(s).

See the doc for tf.random_gamma for further details on sampling strategy.

##### Args:


*  <b>`n`</b>: Python integer, the number of observations to sample from each
    distribution.
*  <b>`seed`</b>: Python integer, the random seed for this operation.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.InverseGamma.std(name='std')` {#InverseGamma.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.InverseGamma.validate_args` {#InverseGamma.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.InverseGamma.variance(name='variance')` {#InverseGamma.variance}

Variance of each batch member.

Variance for inverse gamma is defined only for `alpha > 2`. If
`self.allow_nan_stats` is `False`, an exception will be raised rather
than returning `NaN`.

##### Args:


*  <b>`name`</b>: A name to give this op.

##### Returns:

  The variance for every batch member, a `Tensor` with same `dtype` as self.


