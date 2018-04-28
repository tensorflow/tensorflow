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
"""Vectorized Exponential distribution class, directly using LinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import exponential
from tensorflow.python.ops.distributions import transformed_distribution
from tensorflow.python.ops.linalg import linalg

__all__ = ["VectorExponentialLinearOperator"]

_mvn_sample_note = """
`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

"""


class VectorExponentialLinearOperator(
    transformed_distribution.TransformedDistribution):
  """The vectorization of the Exponential distribution on `R^k`.

  The vector exponential distribution is defined over a subset of `R^k`, and
  parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
  `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is

  ```none
  pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in S(loc, scale),
  x = inv(scale) @ (y - loc),
  Z = |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `S = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`, is an image of
     the positive half-space,
  * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`,
  * `Z` denotes the normalization constant.

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
  mat = [[1.0, 0.1],
         [0.1, 1.0]]

  vex = tfd.VectorExponentialLinearOperator(
      scale=tf.linalg.LinearOperatorFullMatrix(mat))

  # Compute the pdf of an`R^2` observation; return a scalar.
  vex.prob([1., 2.]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Vector Exponential's.
  mu = [[1., 2, 3],
        [1., 0, 0]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  vex = tfd.VectorExponentialLinearOperator(
      loc=mu,
      scale=tf.linalg.LinearOperatorDiag(scale_diag))

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[1.9, 2.2, 3.1],
       [10., 1.0, 9.0]]     # shape: [2, 3]
  vex.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorExponentialLinearOperator"):
    """Construct Vector Exponential distribution supported on a subset of `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale: Instance of `LinearOperator` with same `dtype` as `loc` and shape
        `[B1, ..., Bb, k, k]`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError: if `scale` is unspecified.
      TypeError: if not `scale.dtype.is_floating`
    """
    parameters = locals()
    if scale is None:
      raise ValueError("Missing required `scale` parameter.")
    if not scale.dtype.is_floating:
      raise TypeError("`scale` parameter must have floating-point dtype.")

    with ops.name_scope(name, values=[loc] + scale.graph_parents) as name:
      # Since expand_dims doesn't preserve constant-ness, we obtain the
      # non-dynamic value if possible.
      loc = ops.convert_to_tensor(loc, name="loc") if loc is not None else loc
      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, scale)

      super(VectorExponentialLinearOperator, self).__init__(
          distribution=exponential.Exponential(rate=array_ops.ones(
              [], dtype=scale.dtype), allow_nan_stats=allow_nan_stats),
          bijector=bijectors.AffineLinearOperator(
              shift=loc, scale=scale, validate_args=validate_args),
          batch_shape=batch_shape,
          event_shape=event_shape,
          validate_args=validate_args,
          name=name)
      self._parameters = parameters

  @property
  def loc(self):
    """The `loc` `Tensor` in `Y = scale @ X + loc`."""
    return self.bijector.shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + loc`."""
    return self.bijector.scale

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _log_prob(self, x):
    return super(VectorExponentialLinearOperator, self)._log_prob(x)

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _prob(self, x):
    return super(VectorExponentialLinearOperator, self)._prob(x)

  def _mean(self):
    # Let
    #   W = (w1,...,wk), with wj ~ iid Exponential(0, 1).
    # Then this distribution is
    #   X = loc + LW,
    # and then E[X] = loc + L1, where 1 is the vector of ones.
    scale_x_ones = self.bijector.scale.matvec(
        array_ops.ones(self._mode_mean_shape(), self.dtype))

    if self.loc is None:
      return scale_x_ones

    return array_ops.identity(self.loc) + scale_x_ones

  def _covariance(self):
    # Let
    #   W = (w1,...,wk), with wj ~ iid Exponential(0, 1).
    # Then this distribution is
    #   X = loc + LW,
    # and then since Cov(wi, wj) = 1 if i=j, and 0 otherwise,
    #   Cov(X) = L Cov(W W^T) L^T = L L^T.
    if distribution_util.is_diagonal_scale(self.scale):
      return array_ops.matrix_diag(math_ops.square(self.scale.diag_part()))
    else:
      return self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)

  def _variance(self):
    if distribution_util.is_diagonal_scale(self.scale):
      return math_ops.square(self.scale.diag_part())
    elif (isinstance(self.scale, linalg.LinearOperatorLowRankUpdate) and
          self.scale.is_self_adjoint):
      return array_ops.matrix_diag_part(
          self.scale.matmul(self.scale.to_dense()))
    else:
      return array_ops.matrix_diag_part(
          self.scale.matmul(self.scale.to_dense(), adjoint_arg=True))

  def _stddev(self):
    if distribution_util.is_diagonal_scale(self.scale):
      return math_ops.abs(self.scale.diag_part())
    elif (isinstance(self.scale, linalg.LinearOperatorLowRankUpdate) and
          self.scale.is_self_adjoint):
      return math_ops.sqrt(
          array_ops.matrix_diag_part(self.scale.matmul(self.scale.to_dense())))
    else:
      return math_ops.sqrt(
          array_ops.matrix_diag_part(
              self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)))

  def _mode(self):
    scale_x_zeros = self.bijector.scale.matvec(
        array_ops.zeros(self._mode_mean_shape(), self.dtype))

    if self.loc is None:
      return scale_x_zeros

    return array_ops.identity(self.loc) + scale_x_zeros

  def _mode_mean_shape(self):
    """Shape for the mode/mean Tensors."""
    shape = self.batch_shape.concatenate(self.event_shape)
    has_static_shape = shape.is_fully_defined()
    if not has_static_shape:
      shape = array_ops.concat([
          self.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], 0)
    return shape
