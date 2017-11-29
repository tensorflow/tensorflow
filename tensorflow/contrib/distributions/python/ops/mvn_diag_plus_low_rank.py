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
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import mvn_linear_operator as mvn_linop
from tensorflow.python.framework import ops


__all__ = [
    "MultivariateNormalDiagPlusLowRank",
]


class MultivariateNormalDiagPlusLowRank(
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
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  A (non-batch) `scale` matrix is:

  ```none
  scale = diag(scale_diag + scale_identity_multiplier ones(k)) +
        scale_perturb_factor @ diag(scale_perturb_diag) @ scale_perturb_factor.T
  ```

  where:

  * `scale_diag.shape = [k]`,
  * `scale_identity_multiplier.shape = []`,
  * `scale_perturb_factor.shape = [k, r]`, typically `k >> r`, and,
  * `scale_perturb_diag.shape = [r]`.

  Additional leading dimensions (if any) will index batches.

  If both `scale_diag` and `scale_identity_multiplier` are `None`, then
  `scale` is the Identity matrix.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  ds = tf.contrib.distributions

  # Initialize a single 3-variate Gaussian with covariance `cov = S @ S.T`,
  # `S = diag(d) + U @ diag(m) @ U.T`. The perturbation, `U @ diag(m) @ U.T`, is
  # a rank-2 update.
  mu = [-0.5., 0, 0.5]   # shape: [3]
  d = [1.5, 0.5, 2]      # shape: [3]
  U = [[1., 2],
       [-1, 1],
       [2, -0.5]]        # shape: [3, 2]
  m = [4., 5]            # shape: [2]
  mvn = ds.MultivariateNormalDiagPlusLowRank(
      loc=mu
      scale_diag=d
      scale_perturb_factor=U,
      scale_perturb_diag=m)

  # Evaluate this on an observation in `R^3`, returning a scalar.
  mvn.prob([-1, 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians; `S = diag(d) + U @ U.T`.
  mu = [[1.,  2,  3],
        [11, 22, 33]]      # shape: [b, k] = [2, 3]
  U = [[[1., 2],
        [3,  4],
        [5,  6]],
       [[0.5, 0.75],
        [1,0, 0.25],
        [1.5, 1.25]]]      # shape: [b, k, r] = [2, 3, 2]
  m = [[0.1, 0.2],
       [0.4, 0.5]]         # shape: [b, r] = [2, 2]

  mvn = ds.MultivariateNormalDiagPlusLowRank(
      loc=mu,
      scale_perturb_factor=U,
      scale_perturb_diag=m)

  mvn.covariance().eval()   # shape: [2, 3, 3]
  # ==> [[[  15.63   31.57    48.51]
  #       [  31.57   69.31   105.05]
  #       [  48.51  105.05   162.59]]
  #
  #      [[   2.59    1.41    3.35]
  #       [   1.41    2.71    3.34]
  #       [   3.35    3.34    8.35]]]

  # Compute the pdf of two `R^3` observations (one from each batch);
  # return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               scale_perturb_factor=None,
               scale_perturb_diag=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalDiagPlusLowRank"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

    ```none
    scale = diag(scale_diag + scale_identity_multiplier ones(k)) +
        scale_perturb_factor @ diag(scale_perturb_diag) @ scale_perturb_factor.T
    ```

    where:

    * `scale_diag.shape = [k]`,
    * `scale_identity_multiplier.shape = []`,
    * `scale_perturb_factor.shape = [k, r]`, typically `k >> r`, and,
    * `scale_perturb_diag.shape = [r]`.

    Additional leading dimensions (if any) will index batches.

    If both `scale_diag` and `scale_identity_multiplier` are `None`, then
    `scale` is the Identity matrix.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_diag: Non-zero, floating-point `Tensor` representing a diagonal
        matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
        and characterizes `b`-batches of `k x k` diagonal matrices added to
        `scale`. When both `scale_identity_multiplier` and `scale_diag` are
        `None` then `scale` is the `Identity`.
      scale_identity_multiplier: Non-zero, floating-point `Tensor` representing
        a scaled-identity-matrix added to `scale`. May have shape
        `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
        `k x k` identity matrices added to `scale`. When both
        `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
        the `Identity`.
      scale_perturb_factor: Floating-point `Tensor` representing a rank-`r`
        perturbation added to `scale`. May have shape `[B1, ..., Bb, k, r]`,
        `b >= 0`, and characterizes `b`-batches of rank-`r` updates to `scale`.
        When `None`, no rank-`r` update is added to `scale`.
      scale_perturb_diag: Floating-point `Tensor` representing a diagonal matrix
        inside the rank-`r` perturbation added to `scale`. May have shape
        `[B1, ..., Bb, r]`, `b >= 0`, and characterizes `b`-batches of `r x r`
        diagonal matrices inside the perturbation added to `scale`. When
        `None`, an identity matrix is used inside the perturbation. Can only be
        specified if `scale_perturb_factor` is also specified.
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
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = locals()
    def _convert_to_tensor(x, name):
      return None if x is None else ops.convert_to_tensor(x, name=name)
    with ops.name_scope(name):
      with ops.name_scope("init", values=[
          loc, scale_diag, scale_identity_multiplier, scale_perturb_factor,
          scale_perturb_diag]):
        has_low_rank = (scale_perturb_factor is not None or
                        scale_perturb_diag is not None)
        scale = distribution_util.make_diag_scale(
            loc=loc,
            scale_diag=scale_diag,
            scale_identity_multiplier=scale_identity_multiplier,
            validate_args=validate_args,
            assert_positive=has_low_rank)
        scale_perturb_factor = _convert_to_tensor(
            scale_perturb_factor,
            name="scale_perturb_factor")
        scale_perturb_diag = _convert_to_tensor(
            scale_perturb_diag,
            name="scale_perturb_diag")
        if has_low_rank:
          scale = linalg.LinearOperatorUDVHUpdate(
              scale,
              u=scale_perturb_factor,
              diag_update=scale_perturb_diag,
              is_diag_update_positive=scale_perturb_diag is None,
              is_non_singular=True,  # Implied by is_positive_definite=True.
              is_self_adjoint=True,
              is_positive_definite=True,
              is_square=True)
    super(MultivariateNormalDiagPlusLowRank, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters
