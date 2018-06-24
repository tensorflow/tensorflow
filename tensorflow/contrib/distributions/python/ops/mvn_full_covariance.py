# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multivariate Normal distribution class initialized with a full covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import mvn_tril
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops


__all__ = [
    "MultivariateNormalFullCovariance",
]


class MultivariateNormalFullCovariance(mvn_tril.MultivariateNormalTriL):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `covariance_matrix` matrices that are the covariance.
  This is different than the other multivariate normals, which are parameterized
  by a matrix more akin to the standard deviation.

  #### Mathematical Details

  The probability density function (pdf) is, with `@` as matrix multiplication,

  ```none
  pdf(x; loc, covariance_matrix) = exp(-0.5 y) / Z,
  y = (x - loc)^T @ inv(covariance_matrix) @ (x - loc)
  Z = (2 pi)**(0.5 k) |det(covariance_matrix)|**(0.5).
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `covariance_matrix` is an `R^{k x k}` symmetric positive definite matrix,
  * `Z` denotes the normalization constant.

  Additional leading dimensions (if any) in `loc` and `covariance_matrix` allow
  for batch dimensions.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed e.g. as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  scale = Cholesky(covariance_matrix)
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  mvn = tfd.MultivariateNormalFullCovariance(
      loc=mu,
      covariance_matrix=cov)

  mvn.mean().eval()
  # ==> [1., 2, 3]

  # Covariance agrees with covariance_matrix.
  mvn.covariance().eval()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an observation in `R^3` ; return a scalar.
  mvn.prob([-1., 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  covariance_matrix = ...  # shape: [2, 3, 3], symmetric, positive definite.
  mvn = tfd.MultivariateNormalFullCovariance(
      loc=mu,
      covariance=covariance_matrix)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]

  ```

  """

  def __init__(self,
               loc=None,
               covariance_matrix=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalFullCovariance"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and
    `covariance_matrix` arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `covariance_matrix`. The last dimension of `loc` (if provided) must
    broadcast with this.

    A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
    definite matrix.  In other words it is (real) symmetric with all eigenvalues
    strictly positive.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      covariance_matrix: Floating-point, symmetric positive definite `Tensor` of
        same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
        is ignored, so if `covariance_matrix` is not symmetric no error will be
        raised (unless `validate_args is True`).  `covariance_matrix` has shape
        `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if neither `loc` nor `covariance_matrix` are specified.
    """
    parameters = dict(locals())

    # Convert the covariance_matrix up to a scale_tril and call MVNTriL.
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=[loc, covariance_matrix]):
        if covariance_matrix is None:
          scale_tril = None
        else:
          covariance_matrix = ops.convert_to_tensor(
              covariance_matrix, name="covariance_matrix")
          if validate_args:
            covariance_matrix = control_flow_ops.with_dependencies([
                check_ops.assert_near(
                    covariance_matrix,
                    array_ops.matrix_transpose(covariance_matrix),
                    message="Matrix was not symmetric")], covariance_matrix)
          # No need to validate that covariance_matrix is non-singular.
          # LinearOperatorLowerTriangular has an assert_non_singular method that
          # is called by the Bijector.
          # However, cholesky() ignores the upper triangular part, so we do need
          # to separately assert symmetric.
          scale_tril = linalg_ops.cholesky(covariance_matrix)
        super(MultivariateNormalFullCovariance, self).__init__(
            loc=loc,
            scale_tril=scale_tril,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
    self._parameters = parameters
