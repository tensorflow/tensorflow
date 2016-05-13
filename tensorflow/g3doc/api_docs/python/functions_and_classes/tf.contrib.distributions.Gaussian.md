The scalar Gaussian distribution with mean and stddev parameters mu, sigma.

#### Mathematical details

The PDF of this distribution is:

```f(x) = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))```

#### Examples

Examples of initialization of one or a batch of distributions.

```python
# Define a single scalar Gaussian distribution.
dist = tf.contrib.Gaussian(mu=0, sigma=3)

# Evaluate the cdf at 1, returning a scalar.
dist.cdf(1)

# Define a batch of two scalar valued Gaussians.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tf.contrib.distributions.Gaussian(mu=[1, 2.], sigma=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
dist.pdf([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
dist.sample(3)
```

Arguments are broadcast when possible.

```python
# Define a batch of two scalar valued Gaussians.
# Both have mean 1, but different standard deviations.
dist = tf.contrib.distributions.Gaussian(mu=1, sigma=[11, 22.])

# Evaluate the pdf of both distributions on the same point, 3.0,
# returning a length 2 tensor.
dist.pdf(3.0)
```
- - -

#### `tf.contrib.distributions.Gaussian.__init__(mu, sigma, name=None)` {#Gaussian.__init__}

Construct Gaussian distributions with mean and stddev `mu` and `sigma`.

The parameters `mu` and `sigma` must be shaped in a way that supports
broadcasting (e.g. `mu + sigma` is a valid operation).

##### Args:


*  <b>`mu`</b>: `float` or `double` tensor, the means of the distribution(s).
*  <b>`sigma`</b>: `float` or `double` tensor, the stddevs of the distribution(s).
    sigma must contain only positive values.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if mu and sigma are different dtypes.


- - -

#### `tf.contrib.distributions.Gaussian.cdf(x, name=None)` {#Gaussian.cdf}

CDF of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.dtype` {#Gaussian.dtype}




- - -

#### `tf.contrib.distributions.Gaussian.entropy(name=None)` {#Gaussian.entropy}

The entropy of Gaussian distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Gaussian.log_cdf(x, name=None)` {#Gaussian.log_cdf}

Log CDF of observations `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.log_pdf(x, name=None)` {#Gaussian.log_pdf}

Log pdf of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.mean` {#Gaussian.mean}




- - -

#### `tf.contrib.distributions.Gaussian.mu` {#Gaussian.mu}




- - -

#### `tf.contrib.distributions.Gaussian.pdf(x, name=None)` {#Gaussian.pdf}

The PDF of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: tensor of dtype `dtype`, the pdf values of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.sample(n, seed=None, name=None)` {#Gaussian.sample}

Sample `n` observations from the Gaussian Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.Gaussian.sigma` {#Gaussian.sigma}




