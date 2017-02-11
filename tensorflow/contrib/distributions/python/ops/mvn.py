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
from tensorflow.contrib.distributions.python.ops import bijector as bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.distributions.python.ops import normal
from tensorflow.contrib.distributions.python.ops import transformed_distribution
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


__all__ = [
    "MultivariateNormalDiag",
    "MultivariateNormalDiagWithSoftplusScale",
    "MultivariateNormalDiagPlusLowRank",
    "MultivariateNormalTriL",
]

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


def _convert_to_tensor(x, name):
  """Helper; same as `ops.convert_to_tensor` but passes-through `None`."""
  return x if x is None else ops.convert_to_tensor(x, name=name)


def _event_size_from_loc(loc):
  """Helper; returns the shape of the last dimension of `loc`."""
  # Since tf.gather isn't "constant-in, constant-out", we must first check the
  # static shape or fallback to dynamic shape.
  num_rows = loc.get_shape().with_rank_at_least(1)[-1].value
  if num_rows is not None:
    return num_rows
  return array_ops.shape(loc)[-1]


def _make_diag_scale(loc, scale_diag, scale_identity_multiplier,
                     validate_args, assert_positive):
  """Creates a LinOp from `scale_diag`, `scale_identity_multiplier` kwargs."""
  loc = _convert_to_tensor(loc, name="loc")
  scale_diag = _convert_to_tensor(scale_diag, name="scale_diag")
  scale_identity_multiplier = _convert_to_tensor(
      scale_identity_multiplier,
      name="scale_identity_multiplier")

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return control_flow_ops.with_dependencies([
          check_ops.assert_positive(
              x, message="diagonal part must be positive"),
      ], x)
    # TODO(b/35157376): Use `assert_none_equal` once it exists.
    return control_flow_ops.with_dependencies([
        check_ops.assert_greater(
            math_ops.abs(x),
            array_ops.zeros([], x.dtype),
            message="diagonal part must be non-zero"),
    ], x)

  if scale_diag is not None:
    if scale_identity_multiplier is not None:
      scale_diag += scale_identity_multiplier[..., array_ops.newaxis]
    return linalg.LinearOperatorDiag(
        diag=_maybe_attach_assertion(scale_diag),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive)

  # TODO(b/34878297): Consider inferring shape from scale_perturb_factor.
  if loc is None:
    raise ValueError(
        "Cannot infer `event_shape` unless `loc` is specified.")

  num_rows = _event_size_from_loc(loc)

  if scale_identity_multiplier is None:
    return linalg.LinearOperatorIdentity(
        num_rows=num_rows,
        dtype=loc.dtype.base_dtype,
        is_self_adjoint=True,
        is_positive_definite=True,
        assert_proper_shapes=validate_args)

  return linalg.LinearOperatorScaledIdentity(
      num_rows=num_rows,
      multiplier=_maybe_attach_assertion(scale_identity_multiplier),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=assert_positive,
      assert_proper_shapes=validate_args)


class _MultivariateNormalLinearOperator(
    transformed_distribution.TransformedDistribution):
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
  la = tf.contrib.linalg

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])

  mvn = ds._MultivariateNormalLinearOperator(
      loc=mu,
      scale=la.LinearOperatorTriL(scale))

  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance().eval()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an`R^3` observation; return a scalar.
  mvn.prob([-1., 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  mvn = ds._MultivariateNormalLinearOperator(
      loc=mu,
      scale=la.LinearOperatorDiag(scale_diag))

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalLinearOperator"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by the last dimension of `loc` or the last
    dimension of the matrix implied by `scale`.

    Recall that `covariance = scale @ scale.T`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` represents the event size.
      scale: Instance of `LinearOperator` with same `dtype` as `loc` and shape
        `[B1, ..., Bb, k, k]`.
      validate_args: `Boolean`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`. If `False`, raise an
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

    # Since expand_dims doesn't preserve constant-ness, we obtain the
    # non-dynamic value if possible.
    event_shape = scale.domain_dimension_tensor()
    if tensor_util.constant_value(event_shape) is not None:
      event_shape = tensor_util.constant_value(event_shape)
    event_shape = event_shape[array_ops.newaxis]

    super(_MultivariateNormalLinearOperator, self).__init__(
        distribution=normal.Normal(
            loc=array_ops.zeros([], dtype=scale.dtype),
            scale=array_ops.ones([], dtype=scale.dtype)),
        bijector=bijectors.AffineLinearOperator(
            shift=loc, scale=scale, validate_args=validate_args),
        batch_shape=scale.batch_shape_tensor(),
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

  def log_det_covariance(self, name="log_det_covariance"):
    """Log of determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.scale.graph_parents):
        return 2. * self.scale.log_abs_determinant()

  def det_covariance(self, name="det_covariance"):
    """Determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=self.scale.graph_parents):
        return math_ops.exp(2.* self.scale.log_abs_determinant())

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _log_prob(self, x):
    return super(_MultivariateNormalLinearOperator, self)._log_prob(x)

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _prob(self, x):
    return super(_MultivariateNormalLinearOperator, self)._prob(x)

  def _mean(self):
    if self.loc is None:
      shape = array_ops.concat([
          self.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], 0)
      return array_ops.zeros(shape, self.dtype)
    return array_ops.identity(self.loc)

  def _covariance(self):
    # TODO(b/35041434): Remove special-case logic once LinOp supports
    # `diag_part`.
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      shape = array_ops.concat([self.batch_shape_tensor(),
                                self.event_shape_tensor()], 0)
      diag_part = array_ops.ones(shape, self.scale.dtype)
      if isinstance(self.scale, linalg.LinearOperatorScaledIdentity):
        diag_part *= math_ops.square(
            self.scale.multiplier[..., array_ops.newaxis])
      elif isinstance(self.scale, linalg.LinearOperatorDiag):
        diag_part *= math_ops.square(self.scale.diag)
      return array_ops.matrix_diag(diag_part)
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense()))

  def _variance(self):
    # TODO(b/35041434): Remove special-case logic once LinOp supports
    # `diag_part`.
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      shape = array_ops.concat([self.batch_shape_tensor(),
                                self.event_shape_tensor()], 0)
      diag_part = array_ops.ones(shape, self.scale.dtype)
      if isinstance(self.scale, linalg.LinearOperatorScaledIdentity):
        diag_part *= math_ops.square(
            self.scale.multiplier[..., array_ops.newaxis])
      elif isinstance(self.scale, linalg.LinearOperatorDiag):
        diag_part *= math_ops.square(self.scale.diag)
      return diag_part
    elif (isinstance(self.scale, linalg.LinearOperatorUDVHUpdate)
          and self.scale.is_self_adjoint):
      return array_ops.matrix_diag_part(
          self.scale.apply(self.scale.to_dense()))
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return array_ops.matrix_diag_part(
          self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense())))

  def _stddev(self):
    # TODO(b/35041434): Remove special-case logic once LinOp supports
    # `diag_part`.
    if (isinstance(self.scale, linalg.LinearOperatorIdentity) or
        isinstance(self.scale, linalg.LinearOperatorScaledIdentity) or
        isinstance(self.scale, linalg.LinearOperatorDiag)):
      shape = array_ops.concat([self.batch_shape_tensor(),
                                self.event_shape_tensor()], 0)
      diag_part = array_ops.ones(shape, self.scale.dtype)
      if isinstance(self.scale, linalg.LinearOperatorScaledIdentity):
        diag_part *= self.scale.multiplier[..., array_ops.newaxis]
      elif isinstance(self.scale, linalg.LinearOperatorDiag):
        diag_part *= self.scale.diag
      return math_ops.abs(diag_part)
    elif (isinstance(self.scale, linalg.LinearOperatorUDVHUpdate)
          and self.scale.is_self_adjoint):
      return math_ops.sqrt(array_ops.matrix_diag_part(
          self.scale.apply(self.scale.to_dense())))
    else:
      # TODO(b/35040238): Remove transpose once LinOp supports `transpose`.
      return math_ops.sqrt(array_ops.matrix_diag_part(
          self.scale.apply(array_ops.matrix_transpose(self.scale.to_dense()))))

  def _mode(self):
    return self._mean()


class MultivariateNormalDiagPlusLowRank(_MultivariateNormalLinearOperator):
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

    The `event_shape` is given by the last dimension of `loc` or the last
    dimension of the matrix implied by `scale`.

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
        `b >= 0` and `k` represents the event size.
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
        diagonal matrices inside the perturbation added to `scale`.  When
        `None`, an identity matrix is used inside the perturbation. Can only be
        specified if `scale_perturb_factor` is also specified.
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `Boolean`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: `String` name prefixed to Ops created by this class.

    Raises:
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = locals()
    with ops.name_scope(name) as ns:
      with ops.name_scope("init", values=[
          loc, scale_diag, scale_identity_multiplier, scale_perturb_factor,
          scale_perturb_diag]):
        has_low_rank = (scale_perturb_factor is not None or
                        scale_perturb_diag is not None)
        scale = _make_diag_scale(
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
              diag=scale_perturb_diag,
              is_diag_positive=scale_perturb_diag is None,
              is_non_singular=True,  # Implied by is_positive_definite=True.
              is_self_adjoint=True,
              is_positive_definite=True,
              is_square=True)
    super(MultivariateNormalDiagPlusLowRank, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=ns)
    self._parameters = parameters


class MultivariateNormalDiag(_MultivariateNormalLinearOperator):
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
  scale = diag(scale_diag + scale_identity_multiplier * ones(k))
  ```

  where:

  * `scale_diag.shape = [k]`, and,
  * `scale_identity_multiplier.shape = []`.

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

  # Initialize a single 2-variate Gaussian.
  mvn = ds.MultivariateNormalDiag(
      loc=[1., -1],
      scale_diag=[1, 2.])

  mvn.mean().eval()
  # ==> [1., -1]

  mvn.stddev().eval()
  # ==> [1., 2]

  # Evaluate this on an observation in `R^2`, returning a scalar.
  mvn.prob([-1., 0]).eval()  # shape: []

  # Initialize a 3-batch, 2-variate scaled-identity Gaussian.
  mvn = ds.MultivariateNormalDiag(
      loc=[1., -1],
      scale_identity_multiplier=[1, 2., 3])

  mvn.mean().eval()  # shape: [3, 2]
  # ==> [[1., -1]
  #      [1, -1],
  #      [1, -1]]

  mvn.stddev().eval()  # shape: [3, 2]
  # ==> [[1., 1],
  #      [2, 2],
  #      [3, 3]]

  # Evaluate this on an observation in `R^2`, returning a length-3 vector.
  mvn.prob([-1., 0]).eval()  # shape: [3]

  # Initialize a 2-batch of 3-variate Gaussians.
  mvn = ds.MultivariateNormalDiag(
      loc=[[1., 2, 3],
           [11, 22, 33]]           # shape: [2, 3]
      scale_diag=[[1., 2, 3],
                  [0.5, 1, 1.5]])  # shape: [2, 3]

  # Evaluate this on a two observations, each in `R^3`, returning a length-2
  # vector.
  x = [[-1., 0, 1],
       [-11, 0, 11.]]   # shape: [2, 3].
  mvn.prob(x).eval()    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalDiag"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by the last dimension of `loc` or the last
    dimension of the matrix implied by `scale`.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix
    is:

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
        `b >= 0` and `k` represents the event size.
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
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `Boolean`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: `String` name prefixed to Ops created by this class.

    Raises:
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = locals()
    with ops.name_scope(name) as ns:
      with ops.name_scope("init", values=[
          loc, scale_diag, scale_identity_multiplier]):
        scale = _make_diag_scale(
            loc=loc,
            scale_diag=scale_diag,
            scale_identity_multiplier=scale_identity_multiplier,
            validate_args=validate_args,
            assert_positive=False)
    super(MultivariateNormalDiag, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=ns)
    self._parameters = parameters


class MultivariateNormalDiagWithSoftplusScale(MultivariateNormalDiag):
  """MultivariateNormalDiag with `diag_stddev = softplus(diag_stddev)`."""

  def __init__(self,
               loc,
               scale_diag,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalDiagWithSoftplusScale"):
    parameters = locals()
    with ops.name_scope(name, values=[scale_diag]) as ns:
      super(MultivariateNormalDiagWithSoftplusScale, self).__init__(
          loc=loc,
          scale_diag=nn.softplus(scale_diag),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters


class MultivariateNormalTriL(_MultivariateNormalLinearOperator):
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

  Trainable (batch) Cholesky matrices can be created with
  `ds.matrix_diag_transform()` and/or `ds.fill_lower_triangular()`

  #### Examples

  ```python
  ds = tf.contrib.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  mvn = ds.MultivariateNormalTriL(
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
  mvn = ds.MultivariateNormalTriL(
      loc=mu,
      scale_tril=tril)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]

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

    The `event_shape` is given by the last dimension of `loc` or the last
    dimension of the matrix implied by `scale`.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix
    is:

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
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `Boolean`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: `String` name prefixed to Ops created by this class.

    Raises:
      ValueError: if neither `loc` nor `scale_tril` are specified.
    """
    parameters = locals()
    if loc is None and scale_tril is None:
      raise ValueError("Must specify one or both of `loc`, `scale_tril`.")
    with ops.name_scope(name) as ns:
      with ops.name_scope("init", values=[loc, scale_tril]):
        loc = _convert_to_tensor(loc, name="loc")
        scale_tril = _convert_to_tensor(scale_tril, name="scale_tril")
        if scale_tril is None:
          scale = linalg.LinearOperatorIdentity(
              num_rows=_event_size_from_loc(loc),
              dtype=loc.dtype,
              is_self_adjoint=True,
              is_positive_definite=True,
              assert_proper_shapes=validate_args)
        else:
          if validate_args:
            scale_tril = control_flow_ops.with_dependencies([
                # TODO(b/35157376): Use `assert_none_equal` once it exists.
                check_ops.assert_greater(
                    math_ops.abs(array_ops.matrix_diag_part(scale_tril)),
                    array_ops.zeros([], scale_tril.dtype),
                    message="`scale_tril` must have non-zero diagonal"),
            ], scale_tril)
          scale = linalg.LinearOperatorTriL(
              scale_tril,
              is_non_singular=True,
              is_self_adjoint=False,
              is_positive_definite=False)
    super(MultivariateNormalTriL, self).__init__(
        loc=loc,
        scale=scale,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=ns)
    self._parameters = parameters


@kullback_leibler.RegisterKL(_MultivariateNormalLinearOperator,
                             _MultivariateNormalLinearOperator)
def _kl_brute_force(a, b, name=None):
  """Batched KL divergence `KL(a || b)` for multivariate Normals.

  With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
  covariance `C_a`, `C_b` respectively,

  ```
  KL(a || b) = 0.5 * ( L - k + T + Q ),
  L := Log[Det(C_b)] - Log[Det(C_a)]
  T := trace(C_b^{-1} C_a),
  Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
  ```

  This `Op` computes the trace by solving `C_b^{-1} C_a`. Although efficient
  methods for solving systems with `C_b` may be available, a dense version of
  (the square root of) `C_a` is used, so performance is `O(B s k^2)` where `B`
  is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
  and `y`.

  Args:
    a: Instance of `_MultivariateNormalLinearOperator`.
    b: Instance of `_MultivariateNormalLinearOperator`.
    name: (optional) name to use for created ops. Default "kl_mvn".

  Returns:
    Batchwise `KL(a || b)`.
  """

  def squared_frobenius_norm(x):
    """Helper to make KL calculation slightly more readable."""
    # http://mathworld.wolfram.com/FrobeniusNorm.html
    return math_ops.square(linalg_ops.norm(x, ord="fro", axis=[-2, -1]))

  # TODO(b/35041439): See also b/35040945. Remove this function once LinOp
  # supports something like:
  #   A.inverse().solve(B).norm(order='fro', axis=[-1, -2])
  def is_diagonal(x):
    """Helper to identify if `LinearOperator` has only a diagonal component."""
    return (isinstance(x, linalg.LinearOperatorIdentity) or
            isinstance(x, linalg.LinearOperatorScaledIdentity) or
            isinstance(x, linalg.LinearOperatorDiag))

  with ops.name_scope(name, "kl_mvn", values=[a.loc, b.loc] +
                      a.scale.graph_parents + b.scale.graph_parents):
    # Calculation is based on:
    # http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # and,
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # i.e.,
    #   If Ca = AA', Cb = BB', then
    #   tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
    #                  = tr[inv(B) A A' inv(B)']
    #                  = tr[(inv(B) A) (inv(B) A)']
    #                  = sum_{ij} (inv(B) A)_{ij}^2
    #                  = ||inv(B) A||_F**2
    # where ||.||_F is the Frobenius norm and the second equality follows from
    # the cyclic permutation property.
    if is_diagonal(a.scale) and is_diagonal(b.scale):
      # Using `stddev` because it handles expansion of Identity cases.
      b_inv_a = (a.stddev() / b.stddev())[..., array_ops.newaxis]
    else:
      b_inv_a = b.scale.solve(a.scale.to_dense())
    kl_div = (b.scale.log_abs_determinant()
              - a.scale.log_abs_determinant()
              + 0.5 * (
                  - math_ops.cast(a.scale.domain_dimension_tensor(), a.dtype)
                  + squared_frobenius_norm(b_inv_a)
                  + squared_frobenius_norm(b.scale.solve(
                      (b.mean() - a.mean())[..., array_ops.newaxis]))))
    kl_div.set_shape(array_ops.broadcast_static_shape(
        a.batch_shape, b.batch_shape))
    return kl_div
