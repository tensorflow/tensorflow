A Transformed Distribution.

A Transformed Distribution models `p(y)` given a base distribution `p(x)`,
an invertible transform, `y = f(x)`, and the determinant of the Jacobian of
`f(x)`.

Shapes, type, and reparameterization are taken from the base distribution.

#### Mathematical details

* `p(x)` - probability distribution for random variable X
* `p(y)` - probability distribution for random variable Y
* `f` - transform
* `g` - inverse transform, `g(f(x)) = x`
* `J(x)` - Jacobian of f(x)

A Transformed Distribution exposes `sample` and `pdf`:

  * `sample`: `y = f(x)`, after drawing a sample of X.
  * `pdf`: `p(y) = p(x) / det|J(x)| = p(g(y)) / det|J(g(y))|`

A simple example constructing a Log-Normal distribution from a Normal
distribution:

```
logit_normal = TransformedDistribution(
  base_dist=Normal(mu, sigma),
  transform=lambda x: tf.sigmoid(x),
  inverse=lambda y: tf.log(y) - tf.log(1. - y),
  log_det_jacobian=(lambda x:
      tf.reduce_sum(tf.log(tf.sigmoid(x)) + tf.log(1. - tf.sigmoid(x)),
                    reduction_indices=[-1])))
  name="LogitNormalTransformedDistribution"
)
```
- - -

#### `tf.contrib.distributions.TransformedDistribution.__init__(base_dist_cls, transform, inverse, log_det_jacobian, name='TransformedDistribution', **base_dist_args)` {#TransformedDistribution.__init__}

Construct a Transformed Distribution.

##### Args:


*  <b>`base_dist_cls`</b>: the base distribution class to transform. Must be a
      subclass of `Distribution`.
*  <b>`transform`</b>: a callable that takes a `Tensor` sample from `base_dist` and
      returns a `Tensor` of the same shape and type. `x => y`.
*  <b>`inverse`</b>: a callable that computes the inverse of transform. `y => x`. If
      None, users can only call `log_pdf` on values returned by `sample`.
*  <b>`log_det_jacobian`</b>: a callable that takes a `Tensor` sample from `base_dist`
      and returns the log of the determinant of the Jacobian of `transform`.
*  <b>`name`</b>: The name for the distribution.
*  <b>`**base_dist_args`</b>: kwargs to pass on to dist_cls on construction.

##### Raises:


*  <b>`TypeError`</b>: if `base_dist_cls` is not a subclass of
      `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.allow_nan_stats` {#TransformedDistribution.allow_nan_stats}




- - -

#### `tf.contrib.distributions.TransformedDistribution.base_distribution` {#TransformedDistribution.base_distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.TransformedDistribution.batch_shape(name='batch_shape')` {#TransformedDistribution.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.TransformedDistribution.cdf(value, name='cdf')` {#TransformedDistribution.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.TransformedDistribution.dtype` {#TransformedDistribution.dtype}




- - -

#### `tf.contrib.distributions.TransformedDistribution.entropy(name='entropy')` {#TransformedDistribution.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.TransformedDistribution.event_shape(name='event_shape')` {#TransformedDistribution.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.TransformedDistribution.get_batch_shape()` {#TransformedDistribution.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.TransformedDistribution.get_event_shape()` {#TransformedDistribution.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.TransformedDistribution.inverse` {#TransformedDistribution.inverse}

Inverse function of transform, y => x.


- - -

#### `tf.contrib.distributions.TransformedDistribution.is_continuous` {#TransformedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.TransformedDistribution.is_reparameterized` {#TransformedDistribution.is_reparameterized}




- - -

#### `tf.contrib.distributions.TransformedDistribution.log_cdf(value, name='log_cdf')` {#TransformedDistribution.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_det_jacobian` {#TransformedDistribution.log_det_jacobian}

Function computing the log determinant of the Jacobian of transform.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pdf(value, name='log_pdf')` {#TransformedDistribution.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pmf(value, name='log_pmf')` {#TransformedDistribution.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_prob(y, name='log_prob')` {#TransformedDistribution.log_prob}

Log prob of observations in `y`.

`log ( p(g(y)) / det|J(g(y))| )`, where `g` is the inverse of `transform`.

##### Args:


*  <b>`y`</b>: tensor of dtype `dtype`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `y`.

##### Raises:


*  <b>`ValueError`</b>: if `inverse` was not provided to the distribution and `y` was
      not returned from `sample`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.mean(name='mean')` {#TransformedDistribution.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.TransformedDistribution.mode(name='mode')` {#TransformedDistribution.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.TransformedDistribution.name` {#TransformedDistribution.name}




- - -

#### `tf.contrib.distributions.TransformedDistribution.pdf(value, name='pdf')` {#TransformedDistribution.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.TransformedDistribution.pmf(value, name='pmf')` {#TransformedDistribution.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.TransformedDistribution.prob(y, name='prob')` {#TransformedDistribution.prob}

The prob of observations in `y`.

`p(g(y)) / det|J(g(y))|`, where `g` is the inverse of `transform`.

##### Args:


*  <b>`y`</b>: `Tensor` of dtype `dtype`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: `Tensor` of dtype `dtype`, the pdf values of `y`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.sample(sample_shape=(), seed=None, name='sample')` {#TransformedDistribution.sample}

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

#### `tf.contrib.distributions.TransformedDistribution.sample_n(n, seed=None, name='sample_n')` {#TransformedDistribution.sample_n}

Sample `n` observations.

Samples from the base distribution and then passes through the transform.

##### Args:


*  <b>`n`</b>: scalar, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples.


- - -

#### `tf.contrib.distributions.TransformedDistribution.std(name='std')` {#TransformedDistribution.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.TransformedDistribution.transform` {#TransformedDistribution.transform}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.TransformedDistribution.validate_args` {#TransformedDistribution.validate_args}




- - -

#### `tf.contrib.distributions.TransformedDistribution.variance(name='variance')` {#TransformedDistribution.variance}

Variance of the distribution.


