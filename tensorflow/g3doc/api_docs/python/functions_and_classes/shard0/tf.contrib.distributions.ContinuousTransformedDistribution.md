A Transformed Distribution.

A Transformed Distribution models `p(y)` given a base distribution `p(x)`,
an invertible transform, `y = f(x)`, and the determinant of the Jacobian of
`f(x)`.

Shapes, type, and reparameterization are taken from the base distribution.

#### Mathematical details

* `p(x)` - probability distribution for random variable X
* `p(y)` - probability distribution for random variable Y
* `f` - transform
* `g` - inverse transform, `f(g(x)) = x`
* `J(x)` - Jacobian of f(x)

A Transformed Distribution exposes `sample` and `pdf`:

  * `sample`: `y = f(x)`, after drawing a sample of X.
  * `pdf`: `p(y) = p(x) / det|J(x)| = p(g(y)) / det|J(g(y))|`

A simple example constructing a Log-Normal distribution from a Normal
distribution:

```
logit_normal = ContinuousTransformedDistribution(
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

#### `tf.contrib.distributions.ContinuousTransformedDistribution.__init__(base_dist_cls, transform, inverse, log_det_jacobian, name='ContinuousTransformedDistribution', **base_dist_args)` {#ContinuousTransformedDistribution.__init__}

Construct a Transformed Distribution.

##### Args:


*  <b>`base_dist_cls`</b>: the base distribution class to transform. Must be a
      subclass of `ContinuousDistribution`.
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
      `ContinuousDistribution`.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.base_distribution` {#ContinuousTransformedDistribution.base_distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.batch_shape(name='batch_shape')` {#ContinuousTransformedDistribution.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.cdf(value, name='cdf')` {#ContinuousTransformedDistribution.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.dtype` {#ContinuousTransformedDistribution.dtype}




- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.entropy(name='entropy')` {#ContinuousTransformedDistribution.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.event_shape(name='event_shape')` {#ContinuousTransformedDistribution.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op.

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.get_batch_shape()` {#ContinuousTransformedDistribution.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.get_event_shape()` {#ContinuousTransformedDistribution.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.inverse` {#ContinuousTransformedDistribution.inverse}

Inverse function of transform, y => x.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.is_reparameterized` {#ContinuousTransformedDistribution.is_reparameterized}




- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.log_cdf(value, name='log_cdf')` {#ContinuousTransformedDistribution.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.log_det_jacobian` {#ContinuousTransformedDistribution.log_det_jacobian}

Function computing the log determinant of the Jacobian of transform.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.log_likelihood(value, name='log_likelihood')` {#ContinuousTransformedDistribution.log_likelihood}

Log likelihood of this distribution (same as log_pdf).


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.log_pdf(y, name='log_pdf')` {#ContinuousTransformedDistribution.log_pdf}

Log pdf of observations in `y`.

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

#### `tf.contrib.distributions.ContinuousTransformedDistribution.mean(name='mean')` {#ContinuousTransformedDistribution.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.mode(name='mode')` {#ContinuousTransformedDistribution.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.name` {#ContinuousTransformedDistribution.name}




- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.pdf(y, name='pdf')` {#ContinuousTransformedDistribution.pdf}

The PDF of observations in `y`.

`p(g(y)) / det|J(g(y))|`, where `g` is the inverse of `transform`.

##### Args:


*  <b>`y`</b>: `Tensor` of dtype `dtype`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: `Tensor` of dtype `dtype`, the pdf values of `y`.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.sample(n, seed=None, name='sample')` {#ContinuousTransformedDistribution.sample}

Sample `n` observations.

Samples from the base distribution and then passes through the transform.

##### Args:


*  <b>`n`</b>: scalar, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.std(name='std')` {#ContinuousTransformedDistribution.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.strict` {#ContinuousTransformedDistribution.strict}




- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.strict_statistics` {#ContinuousTransformedDistribution.strict_statistics}




- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.transform` {#ContinuousTransformedDistribution.transform}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.ContinuousTransformedDistribution.variance(name='variance')` {#ContinuousTransformedDistribution.variance}

Variance of the distribution.


