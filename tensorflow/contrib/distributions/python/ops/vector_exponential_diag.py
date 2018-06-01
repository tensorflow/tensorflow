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
"""Distribution of a vectorized Exponential, with uncorrelated components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import vector_exponential_linear_operator as vector_exponential_linop
from tensorflow.python.framework import ops


__all__ = [
    "VectorExponentialDiag",
]


class VectorExponentialDiag(
    vector_exponential_linop.VectorExponentialLinearOperator):
  """The vectorization of the Exponential distribution on `R^k`.

  The vector exponential distribution is defined over a subset of `R^k`, and
  parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
  `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is defined over the image of the
  `scale` matrix + `loc`, applied to the positive half-space:
  `Supp = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`.  On this set,

  ```none
  pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in Supp
  x = inv(scale) @ (y - loc),
  Z = |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`.

  The VectorExponential distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X = (X_1, ..., X_k), each X_i ~ Exponential(rate=1)
  Y = (Y_1, ...,Y_k) = scale @ X + loc
  ```

  #### About `VectorExponential` and `Vector` distributions in TensorFlow.

  The `VectorExponential` is a non-standard distribution that has useful
  properties.

  The marginals `Y_1, ..., Y_k` are *not* Exponential random variables, due to
  the fact that the sum of Exponential random variables is not Exponential.

  Instead, `Y` is a vector whose components are linear combinations of
  Exponential random variables.  Thus, `Y` lives in the vector space generated
  by `vectors` of Exponential distributions.  This allows the user to decide the
  mean and covariance (by setting `loc` and `scale`), while preserving some
  properties of the Exponential distribution.  In particular, the tails of `Y_i`
  will be (up to polynomial factors) exponentially decaying.

  To see this last statement, note that the pdf of `Y_i` is the convolution of
  the pdf of `k` independent Exponential random variables.  One can then show by
  induction that distributions with exponential (up to polynomial factors) tails
  are closed under convolution.


  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Initialize a single 2-variate VectorExponential, supported on
  # {(x, y) in R^2 : x > 0, y > 0}.

  # The first component has pdf exp{-x}, the second 0.5 exp{-x / 2}
  vex = tfd.VectorExponentialDiag(scale_diag=[1., 2.])

  # Compute the pdf of an`R^2` observation; return a scalar.
  vex.prob([3., 4.]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Vector Exponential's.
  loc = [[1., 2, 3],
         [1., 0, 0]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  vex = tfd.VectorExponentialDiag(loc, scale_diag)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[1.9, 2.2, 3.1],
       [10., 1.0, 9.0]]     # shape: [2, 3]
  vex.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorExponentialDiag"):
    """Construct Vector Exponential distribution supported on a subset of `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`.

    ```none
    scale = diag(scale_diag + scale_identity_multiplier * ones(k))
    ```

    where:

    * `scale_diag.shape = [k]`, and,
    * `scale_identity_multiplier.shape = []`.

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
    parameters = dict(locals())
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=[
          loc, scale_diag, scale_identity_multiplier]):
        # No need to validate_args while making diag_scale.  The returned
        # LinearOperatorDiag has an assert_non_singular method that is called by
        # the Bijector.
        scale = distribution_util.make_diag_scale(
            loc=loc,
            scale_diag=scale_diag,
            scale_identity_multiplier=scale_identity_multiplier,
            validate_args=False,
            assert_positive=False)
    super(VectorExponentialDiag, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters
