The Exponential distribution with rate parameter lam.

The PDF of this distribution is:

```pdf(x) = (lam * e^(-lam * x)), x > 0```

Note that the Exponential distribution is a special case of the Gamma
distribution, with Exponential(lam) = Gamma(1, lam).
- - -

#### `tf.contrib.distributions.Exponential.__init__(lam, name='Exponential')` {#Exponential.__init__}




- - -

#### `tf.contrib.distributions.Exponential.alpha` {#Exponential.alpha}




- - -

#### `tf.contrib.distributions.Exponential.batch_shape(name='batch_shape')` {#Exponential.batch_shape}




- - -

#### `tf.contrib.distributions.Exponential.beta` {#Exponential.beta}




- - -

#### `tf.contrib.distributions.Exponential.cdf(x, name='cdf')` {#Exponential.cdf}




- - -

#### `tf.contrib.distributions.Exponential.dtype` {#Exponential.dtype}




- - -

#### `tf.contrib.distributions.Exponential.entropy(name='entropy')` {#Exponential.entropy}

The entropy of Gamma distribution(s).

This is defined to be

```entropy = alpha - log(beta) + log(Gamma(alpha))
             + (1-alpha)digamma(alpha)```

where digamma(alpha) is the digamma function.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Exponential.event_shape(name='event_shape')` {#Exponential.event_shape}




- - -

#### `tf.contrib.distributions.Exponential.get_batch_shape()` {#Exponential.get_batch_shape}




- - -

#### `tf.contrib.distributions.Exponential.get_event_shape()` {#Exponential.get_event_shape}




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

#### `tf.contrib.distributions.Exponential.log_pdf(x, name='log_pdf')` {#Exponential.log_pdf}

Log pdf of observations in `x` under these Gamma distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `alpha` and `beta`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.

##### Raises:


*  <b>`TypeError`</b>: if `x` and `alpha` are different dtypes.


- - -

#### `tf.contrib.distributions.Exponential.mean` {#Exponential.mean}




- - -

#### `tf.contrib.distributions.Exponential.name` {#Exponential.name}




- - -

#### `tf.contrib.distributions.Exponential.pdf(x, name='pdf')` {#Exponential.pdf}




- - -

#### `tf.contrib.distributions.Exponential.sample(n, seed=None, name=None)` {#Exponential.sample}

Sample `n` observations from the Exponential Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by the hyperparameters.


- - -

#### `tf.contrib.distributions.Exponential.variance` {#Exponential.variance}




