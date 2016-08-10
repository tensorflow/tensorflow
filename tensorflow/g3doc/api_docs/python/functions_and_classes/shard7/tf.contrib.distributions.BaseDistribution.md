Simple abstract base class for probability distributions.

Implementations of core distributions to be included in the `distributions`
module should subclass `Distribution`. This base class may be useful to users
that want to fulfill a simpler distribution contract.
- - -

#### `tf.contrib.distributions.BaseDistribution.log_prob(value, name='log_prob')` {#BaseDistribution.log_prob}

Log of the probability density/mass function.


- - -

#### `tf.contrib.distributions.BaseDistribution.name` {#BaseDistribution.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.BaseDistribution.prob(value, name='prob')` {#BaseDistribution.prob}

Probability density/mass function.


- - -

#### `tf.contrib.distributions.BaseDistribution.sample(sample_shape=(), seed=None, name='sample')` {#BaseDistribution.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: int32 `Tensor` or tuple or list. Shape of the generated
    samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.BaseDistribution.sample_n(n, seed=None, name='sample_n')` {#BaseDistribution.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with a prepended dimension (n,).


