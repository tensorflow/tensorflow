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
from tensorflow.contrib.distributions.python.ops import student_t
from tensorflow.contrib.distributions.python.ops import transformed_distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops


# TODO(jvdillon): Add unittests for this once we know where will put this code
# and how it will generally be used. In the interim this code is tested via the
# _VectorStudentT tests.
def _infer_shapes(scale_oppd, shift):
  """Helper which returns batch_shape, event_shape from `Affine` properties.

  The `Affine` `Bijector` (roughly) computes `Y = scale @ X.T + shift`. This
  function infers the `batch_shape` and `event_shape` from the `scale` and
  `shift` terms.

  Args:
    scale_oppd: Instance of OperatorPDBase subclass representing the `Affine`
      `Bijector` scale matrix.
    shift: `Tensor` representing the `shift` vector.

  Returns:
    batch_shape: 1D, integer `Tensor` representing the shape of batch
      dimensions.
    event_shape: 1D, integer `Tensor` representing the shape of event
      dimensions.

  Raises:
    ValueError: if we are not able to infer batch/event shapes from the args.
  """
  # Collect known static shape.
  def _has_static_ndims(x):
    return x is not None and x.get_shape().ndims is not None
  if _has_static_ndims(scale_oppd) and _has_static_ndims(shift):
    batch_shape = scale_oppd.get_batch_shape().merge_with(
        shift.get_shape()[:-1])
    event_shape = scale_oppd.get_shape()[-1:].merge_with(
        shift.get_shape()[-1:])
  elif _has_static_ndims(scale_oppd):
    batch_shape = scale_oppd.get_batch_shape()
    event_shape = scale_oppd.get_shape()[-1:]
  elif _has_static_ndims(shift):
    batch_shape = shift.get_shape()[:-1]
    event_shape = shift.get_shape()[-1:]
  else:
    batch_shape = tensor_shape.TensorShape(None)
    event_shape = tensor_shape.TensorShape(None)

  # Convert TensorShape to Tensors and see if we're done.
  if batch_shape.is_fully_defined():
    batch_shape = constant_op.constant(batch_shape.as_list(),
                                       dtype=dtypes.int32)
  else:
    batch_shape = None
  if event_shape.is_fully_defined():
    event_shape = constant_op.constant(event_shape.as_list(),
                                       dtype=dtypes.int32)
  else:
    event_shape = None
  if batch_shape is not None and event_shape is not None:
    return batch_shape, event_shape

  # Collect known dynamic shape.
  if scale_oppd is not None:
    shape = scale_oppd.shape()
  elif shift is not None:
    shape = array_ops.shape(shift)
  else:
    raise ValueError("unable to infer batch_shape, event_shape")

  # Fill in what we don't know.
  if batch_shape is None:
    batch_shape = array_ops.identity(shape[:-1], name="batch_shape")
  if event_shape is None:
    event_shape = array_ops.identity(shape[-1:], name="event_shape")

  return batch_shape, event_shape


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
  and should not be confused with the [Multivate Student's t-distribution](
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
  vector of of length `k` and a scale matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  ds = tf.contrib.distributions

  # Initialize a single 3-variate vector Student's t-distribution.
  mu = [1., 2, 3]
  chol = [[1., 0, 0.],
          [1, 3, 0],
          [1, 2, 3]]
  vt = ds.VectorStudentT(df=2, loc=mu, scale_tril=chol)

  # Evaluate this on an observation in R^3, returning a scalar.
  vt.prob([-1., 0, 1])

  # Initialize a batch of two 3-variate vector Student's t-distributions.
  mu = [[1., 2, 3],
        [11, 22, 33]]
  chol = ...  # shape 2 x 3 x 3, lower triangular, positive diagonal.
  vt = ds.VectorStudentT(loc=mu, scale_tril=chol)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1],
       [-11, 0, 11]]
  vt.prob(x)
  ```

  For more examples of how to construct the `scale` matrix, see the
  `bijectors.Affine` docstring.

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
    parameters = locals()
    graph_parents = [df, loc, scale_identity_multiplier, scale_diag,
                     scale_tril, scale_perturb_factor, scale_perturb_diag]
    with ops.name_scope(name) as ns:
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
        batch_shape, override_event_shape = _infer_shapes(
            affine.scale, affine.shift)
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
            name=ns)
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
