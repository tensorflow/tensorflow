The Multivariate Normal distribution on `R^k`.

The distribution has mean and covariance parameters mu (1-D), sigma (2-D),
or alternatively mean `mu` and factored covariance (cholesky decomposed
`sigma`) called `sigma_chol`.

#### Mathematical details

The PDF of this distribution is:

```
f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
```

where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

Alternatively, if `sigma` is positive definite, it can be represented in terms
of its lower triangular cholesky factorization

```sigma = sigma_chol . sigma_chol^*```

and the pdf above allows simpler computation:

```
|det(sigma)| = reduce_prod(diag(sigma_chol))^2
x_whitened = sigma^{-1/2} . (x - mu) = tri_solve(sigma_chol, x - mu)
(x-mu)^* .sigma^{-1} . (x-mu) = x_whitened^* . x_whitened
```

where `tri_solve()` solves a triangular system of equations.

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and a covariance matrix of shape `k x k`.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal covariance.
mu = [1, 2, 3]
sigma = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.
mu = [[1, 2, 3], [11, 22, 33]]
sigma = ...  # shape 2 x 3 x 3
dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
dist.pdf(x)
```
- - -

#### `tf.contrib.distributions.MultivariateNormal.__init__(mu, sigma=None, sigma_chol=None, name=None)` {#MultivariateNormal.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu`, which are tensors of rank `N+1` (`N >= 0`)
with the last dimension having length `k`.

User must provide exactly one of `sigma` (the covariance matrices) or
`sigma_chol` (the cholesky decompositions of the covariance matrices).
`sigma` or `sigma_chol` must be of rank `N+2`.  The last two dimensions
must both have length `k`.  The first `N` dimensions correspond to batch
indices.

If `sigma_chol` is not provided, the batch cholesky factorization of `sigma`
is calculated for you.

The shapes of `mu` and `sigma` must match for the first `N` dimensions.

Regardless of which parameter is provided, the covariance matrices must all
be **positive definite** (an error is raised if one of them is not).

##### Args:


*  <b>`mu`</b>: (N+1)-D.  `float` or `double` tensor, the means of the distributions.
*  <b>`sigma`</b>: (N+2)-D.  (optional) `float` or `double` tensor, the covariances
    of the distribution(s).  The first `N+1` dimensions must match
    those of `mu`.  Must be batch-positive-definite.
*  <b>`sigma_chol`</b>: (N+2)-D.  (optional) `float` or `double` tensor, a
    lower-triangular factorization of `sigma`
    (`sigma = sigma_chol . sigma_chol^*`).  The first `N+1` dimensions
    must match those of `mu`.  The tensor itself need not be batch
    lower triangular: we ignore the upper triangular part.  However,
    the batch diagonals must be positive (i.e., sigma_chol must be
    batch-positive-definite).
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`ValueError`</b>: if neither sigma nor sigma_chol is provided.
*  <b>`TypeError`</b>: if mu and sigma (resp. sigma_chol) are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormal.dtype` {#MultivariateNormal.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormal.entropy(name=None)` {#MultivariateNormal.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormal.is_reparameterized` {#MultivariateNormal.is_reparameterized}




- - -

#### `tf.contrib.distributions.MultivariateNormal.log_pdf(x, name=None)` {#MultivariateNormal.log_pdf}

Log pdf of observations `x` given these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormal.mean` {#MultivariateNormal.mean}




- - -

#### `tf.contrib.distributions.MultivariateNormal.mu` {#MultivariateNormal.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormal.pdf(x, name=None)` {#MultivariateNormal.pdf}

The PDF of observations `x` under these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: tensor of dtype `dtype`, the pdf values of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormal.sample(n, seed=None, name=None)` {#MultivariateNormal.sample}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormal.sigma` {#MultivariateNormal.sigma}




- - -

#### `tf.contrib.distributions.MultivariateNormal.sigma_det` {#MultivariateNormal.sigma_det}




