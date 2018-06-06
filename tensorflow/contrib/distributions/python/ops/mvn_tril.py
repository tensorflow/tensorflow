# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Multivariate Normal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import linalg
from tensorflow.contrib.distributions.python.ops import mvn_linear_operator as mvn_linop
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import util as distribution_util


__all__ = [
    "MultivariateNormalTriL",
]


class MultivariateNormalTriL(
    mvn_linop.MultivariateNormalLinearOperator):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `scale` matrix; `covariance = scale @ scale.T` where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
  y = inv(scale) @ (x - loc),
  Z = (2 pi)**(0.5 k) |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a matrix in `R^{k x k}`, `covariance = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  A (non-batch) `scale` matrix is:

  ```none
  scale = scale_tril
  ```

  where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
  i.e., `tf.diag_part(scale_tril) != 0`.

  Additional leading dimensions (if any) will index batches.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  Trainable (batch) lower-triangular matrices can be created with
  `tf.contrib.distributions.matrix_diag_transform()` and/or
  `tf.contrib.distributions.fill_triangular()`

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=scale)

  mvn.mean().eval()
  # ==> [1., 2, 3]

  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance().eval()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an observation in `R^3` ; return a scalar.
  mvn.prob([-1., 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=tril)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]

  # Instantiate a "learnable" MVN.
  dims = 4
  with tf.variable_scope("model"):
    mvn = tfd.MultivariateNormalTriL(
        loc=tf.get_variable(shape=[dims], dtype=tf.float32, name="mu"),
        scale_tril=tfd.fill_triangular(
            tf.get_variable(shape=[dims * (dims + 1) / 2],
                            dtype=tf.float32, name="chol_Sigma")))
  ```

  """

  def __init__(self,
               loc=None,
               scale_tril=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalTriL"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

    ```none
    scale = scale_tril
    ```

    where `scale_tril` is lower-triangular `k x k` matrix with non-zero
    diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_tril: Floating-point, lower-triangular `Tensor` with non-zero
        diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
        `b >= 0` and `k` is the event size.
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
      ValueError: if neither `loc` nor `scale_tril` are specified.
    """
    parameters = distribution_util.parent_frame_arguments()
    def _convert_to_tensor(x, name):
      return None if x is None else ops.convert_to_tensor(x, name=name)
    if loc is None and scale_tril is None:
      raise ValueError("Must specify one or both of `loc`, `scale_tril`.")
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=[loc, scale_tril]):
        loc = _convert_to_tensor(loc, name="loc")
        scale_tril = _convert_to_tensor(scale_tril, name="scale_tril")
        if scale_tril is None:
          scale = linalg.LinearOperatorIdentity(
              num_rows=distribution_util.dimension_size(loc, -1),
              dtype=loc.dtype,
              is_self_adjoint=True,
              is_positive_definite=True,
              assert_proper_shapes=validate_args)
        else:
          # No need to validate that scale_tril is non-singular.
          # LinearOperatorLowerTriangular has an assert_non_singular
          # method that is called by the Bijector.
          scale = linalg.LinearOperatorLowerTriangular(
              scale_tril,
              is_non_singular=True,
              is_self_adjoint=False,
              is_positive_definite=False)
    super(MultivariateNormalTriL, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters
