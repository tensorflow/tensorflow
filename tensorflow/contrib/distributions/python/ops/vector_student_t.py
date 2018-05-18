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
"""Vector Student's t distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import student_t
from tensorflow.python.ops.distributions import transformed_distribution


class _VectorStudentT(transformed_distribution.TransformedDistribution):
  """A vector version of Student's t-distribution on `R^k`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, Sigma) = (1 + ||y||**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = inv(Sigma) (x - mu)
  Z = abs(det(Sigma)) ( sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1)) )**k
  ```

  where:
  * `loc = mu`; a vector in `R^k`,
  * `scale = Sigma`; a lower-triangular matrix in `R^{k x k}`,
  * `Z` denotes the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function), and,
  * `||y||**2` denotes the [squared Euclidean norm](
  https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of `y`.

  The VectorStudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that the `scale` matrix has semantics closer to std. deviation than
  covariance (but it is not std. deviation).

  This distribution is an Affine transformation of iid
  [Student's t-distributions](
  https://en.wikipedia.org/wiki/Student%27s_t-distribution)
  and should not be confused with the [Multivariate Student's t-distribution](
  https://en.wikipedia.org/wiki/Multivariate_t-distribution). The
  traditional Multivariate Student's t-distribution is type of
  [elliptical distribution](
  https://en.wikipedia.org/wiki/Elliptical_distribution); it has PDF:

  ```none
  pdf(x; df, mu, Sigma) = (1 + ||y||**2 / df)**(-0.5 (df + k)) / Z
  where,
  y = inv(Sigma) (x - mu)
  Z = abs(det(Sigma)) sqrt(df pi)**k Gamma(0.5 df) / Gamma(0.5 (df + k))
  ```

  Notice that the Multivariate Student's t-distribution uses `k` where the
  Vector Student's t-distribution has a `1`. Conversely the Vector version has a
  broader application of the power-`k` in the normalization constant.

  #### Examples

  A single instance of a "Vector Student's t-distribution" is defined by a mean
  vector of length `k` and a scale matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  tfd = tf.contrib.distributions

  # Initialize a single 3-variate vector Student's t-distribution.
  mu = [1., 2, 3]
  chol = [[1., 0, 0.],
          [1, 3, 0],
          [1, 2, 3]]
  vt = tfd.VectorStudentT(df=2, loc=mu, scale_tril=chol)

  # Evaluate this on an observation in R^3, returning a scalar.
  vt.prob([-1., 0, 1])

  # Initialize a batch of two 3-variate vector Student's t-distributions.
  mu = [[1., 2, 3],
        [11, 22, 33]]
  chol = ...  # shape 2 x 3 x 3, lower triangular, positive diagonal.
  vt = tfd.VectorStudentT(loc=mu, scale_tril=chol)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1],
       [-11, 0, 11]]
  vt.prob(x)
  ```

  For more examples of how to construct the `scale` matrix, see the
  `tf.contrib.distributions.bijectors.Affine` docstring.

  """

  def __init__(self,
               df,
               loc=None,
               scale_identity_multiplier=None,
               scale_diag=None,
               scale_tril=None,
               scale_perturb_factor=None,
               scale_perturb_diag=None,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorStudentT"):
    """Instantiates the vector Student's t-distributions on `R^k`.

    The `batch_shape` is the broadcast between `df.batch_shape` and
    `Affine.batch_shape` where `Affine` is constructed from `loc` and
    `scale_*` arguments.

    The `event_shape` is the event shape of `Affine.event_shape`.

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values. Must be
        scalar if `loc`, `scale_*` imply non-scalar batch_shape or must have the
        same `batch_shape` implied by `loc`, `scale_*`.
      loc: Floating-point `Tensor`. If this is set to `None`, no `loc` is
        applied.
      scale_identity_multiplier: floating point rank 0 `Tensor` representing a
        scaling done to the identity matrix. When `scale_identity_multiplier =
        scale_diag=scale_tril = None` then `scale += IdentityMatrix`. Otherwise
        no scaled-identity-matrix is added to `scale`.
      scale_diag: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape [N1, N2, ..., k], which represents a k x k
        diagonal matrix. When `None` no diagonal term is added to `scale`.
      scale_tril: Floating-point `Tensor` representing the diagonal matrix.
        `scale_diag` has shape [N1, N2, ..., k, k], which represents a k x k
        lower triangular matrix. When `None` no `scale_tril` term is added to
        `scale`. The upper triangular elements above the diagonal are ignored.
      scale_perturb_factor: Floating-point `Tensor` representing factor matrix
        with last two dimensions of shape `(k, r)`. When `None`, no rank-r
        update is added to `scale`.
      scale_perturb_diag: Floating-point `Tensor` representing the diagonal
        matrix. `scale_perturb_diag` has shape [N1, N2, ..., r], which
        represents an r x r Diagonal matrix. When `None` low rank updates will
        take the form `scale_perturb_factor * scale_perturb_factor.T`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = distribution_util.parent_frame_arguments()
    graph_parents = [df, loc, scale_identity_multiplier, scale_diag,
                     scale_tril, scale_perturb_factor, scale_perturb_diag]
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=graph_parents):
        # The shape of the _VectorStudentT distribution is governed by the
        # relationship between df.batch_shape and affine.batch_shape. In
        # pseudocode the basic procedure is:
        #   if df.batch_shape is scalar:
        #     if affine.batch_shape is not scalar:
        #       # broadcast distribution.sample so
        #       # it has affine.batch_shape.
        #     self.batch_shape = affine.batch_shape
        #   else:
        #     if affine.batch_shape is scalar:
        #       # let affine broadcasting do its thing.
        #     self.batch_shape = df.batch_shape
        # All of the above magic is actually handled by TransformedDistribution.
        # Here we really only need to collect the affine.batch_shape and decide
        # what we're going to pass in to TransformedDistribution's
        # (override) batch_shape arg.
        affine = bijectors.Affine(
            shift=loc,
            scale_identity_multiplier=scale_identity_multiplier,
            scale_diag=scale_diag,
            scale_tril=scale_tril,
            scale_perturb_factor=scale_perturb_factor,
            scale_perturb_diag=scale_perturb_diag,
            validate_args=validate_args)
        distribution = student_t.StudentT(
            df=df,
            loc=array_ops.zeros([], dtype=affine.dtype),
            scale=array_ops.ones([], dtype=affine.dtype))
        batch_shape, override_event_shape = (
            distribution_util.shapes_from_loc_and_scale(
                affine.shift, affine.scale))
        override_batch_shape = distribution_util.pick_vector(
            distribution.is_scalar_batch(),
            batch_shape,
            constant_op.constant([], dtype=dtypes.int32))
        super(_VectorStudentT, self).__init__(
            distribution=distribution,
            bijector=affine,
            batch_shape=override_batch_shape,
            event_shape=override_event_shape,
            validate_args=validate_args,
            name=name)
        self._parameters = parameters

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self.distribution.df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self.bijector.shift

  @property
  def scale(self):
    """Dense (batch) covariance matrix, if available."""
    return self.bijector.scale
