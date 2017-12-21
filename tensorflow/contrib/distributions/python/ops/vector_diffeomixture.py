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
"""The VectorDiffeomixture distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops.bijectors.affine_linear_operator import AffineLinearOperator
from tensorflow.contrib.distributions.python.ops.bijectors.softmax_centered import SoftmaxCentered
from tensorflow.contrib.linalg.python.ops import linear_operator_addition as linop_add_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import categorical as categorical_lib
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.linalg import linear_operator_diag as linop_diag_lib
from tensorflow.python.ops.linalg import linear_operator_full_matrix as linop_full_lib
from tensorflow.python.ops.linalg import linear_operator_identity as linop_identity_lib
from tensorflow.python.ops.linalg import linear_operator_lower_triangular as linop_tril_lib


__all__ = [
    "VectorDiffeomixture",
    "quadrature_scheme_softmaxnormal_gauss_hermite",
    "quadrature_scheme_softmaxnormal_quantiles",
]


def quadrature_scheme_softmaxnormal_gauss_hermite(
    loc, scale, quadrature_size,
    validate_args=False, name=None):
  """Use Gauss-Hermite quadrature to form quadrature on `K - 1` simplex.

  Note: for a given `quadrature_size`, this method is generally less accurate
  than `quadrature_scheme_softmaxnormal_quantiles`.

  Args:
    loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      Represents the `location` parameter of the SoftmaxNormal used for
      selecting one of the `K` affine transformations.
    scale: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      Represents the `scale` parameter of the SoftmaxNormal used for
      selecting one of the `K` affine transformations.
    quadrature_size: Python `int` scalar representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
      convex combination of affine parameters for `K` components.
      `grid[..., :, n]` is the `n`-th grid point, living in the `K - 1` simplex.
    probs:  Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
      associated with each grid point.
  """
  with ops.name_scope(name, "quadrature_scheme_softmaxnormal_gauss_hermite",
                      [loc, scale]):
    loc = ops.convert_to_tensor(loc, name="loc")
    dt = loc.dtype.base_dtype
    scale = ops.convert_to_tensor(scale, dtype=dt, name="scale")

    loc = maybe_check_quadrature_param(loc, "loc", validate_args)
    scale = maybe_check_quadrature_param(scale, "scale", validate_args)

    grid, probs = np.polynomial.hermite.hermgauss(deg=quadrature_size)
    grid = grid.astype(loc.dtype.as_numpy_dtype)
    probs = probs.astype(loc.dtype.as_numpy_dtype)
    probs /= np.linalg.norm(probs, ord=1, keepdims=True)
    probs = ops.convert_to_tensor(probs, name="probs", dtype=loc.dtype)

    grid = softmax(
        -distribution_util.pad(
            (loc[..., array_ops.newaxis] +
             np.sqrt(2.) * scale[..., array_ops.newaxis] * grid),
            axis=-2,
            front=True),
        axis=-2)  # shape: [B, components, deg]

    return grid, probs


def quadrature_scheme_softmaxnormal_quantiles(
    loc, scale, quadrature_size,
    validate_args=False, name=None):
  """Use SoftmaxNormal quantiles to form quadrature on `K - 1` simplex.

  Args:
    loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      Represents the `location` parameter of the SoftmaxNormal used for
      selecting one of the `K` affine transformations.
    scale: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      Represents the `scale` parameter of the SoftmaxNormal used for
      selecting one of the `K` affine transformations.
    quadrature_size: Python scalar `int` representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
      convex combination of affine parameters for `K` components.
      `grid[..., :, n]` is the `n`-th grid point, living in the `K - 1` simplex.
    probs:  Shape `[b1, ..., bB, K, quadrature_size]` `Tensor` representing the
      associated with each grid point.
  """
  with ops.name_scope(name, "softmax_normal_grid_and_probs", [loc, scale]):
    loc = ops.convert_to_tensor(loc, name="loc")
    dt = loc.dtype.base_dtype
    scale = ops.convert_to_tensor(scale, dtype=dt, name="scale")

    loc = maybe_check_quadrature_param(loc, "loc", validate_args)
    scale = maybe_check_quadrature_param(scale, "scale", validate_args)

    dist = normal_lib.Normal(loc=loc, scale=scale)

    def _get_batch_ndims():
      """Helper to get dist.batch_shape.ndims, statically if possible."""
      ndims = dist.batch_shape.ndims
      if ndims is None:
        ndims = array_ops.shape(dist.batch_shape_tensor())[0]
      return ndims
    batch_ndims = _get_batch_ndims()

    def _get_final_shape(qs):
      """Helper to build `TensorShape`."""
      bs = dist.batch_shape.with_rank_at_least(1)
      num_components = bs[-1].value
      if num_components is not None:
        num_components += 1
      tail = tensor_shape.TensorShape([num_components, qs])
      return bs[:-1].concatenate(tail)

    def _compute_quantiles():
      """Helper to build quantiles."""
      # Omit {0, 1} since they might lead to Inf/NaN.
      zero = array_ops.zeros([], dtype=dist.dtype)
      edges = math_ops.linspace(zero, 1., quadrature_size + 3)[1:-1]
      # Expand edges so its broadcast across batch dims.
      edges = array_ops.reshape(edges, shape=array_ops.concat([
          [-1], array_ops.ones([batch_ndims], dtype=dtypes.int32)], axis=0))
      quantiles = dist.quantile(edges)
      quantiles = SoftmaxCentered(event_ndims=1).forward(quantiles)
      # Cyclically permute left by one.
      perm = array_ops.concat([
          math_ops.range(1, 1 + batch_ndims), [0]], axis=0)
      quantiles = array_ops.transpose(quantiles, perm)
      quantiles.set_shape(_get_final_shape(quadrature_size + 1))
      return quantiles
    quantiles = _compute_quantiles()

    # Compute grid as quantile midpoints.
    grid = (quantiles[..., :-1] + quantiles[..., 1:]) / 2.
    # Set shape hints.
    grid.set_shape(_get_final_shape(quadrature_size))

    # By construction probs is constant, i.e., `1 / quadrature_size`. This is
    # important, because non-constant probs leads to non-reparameterizable
    # samples.
    probs = array_ops.fill(
        dims=[quadrature_size],
        value=1. / math_ops.cast(quadrature_size, dist.dtype))

    return grid, probs


class VectorDiffeomixture(distribution_lib.Distribution):
  """VectorDiffeomixture distribution.

  The VectorDiffeomixture is an approximation to a [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e.,

  ```none
  p(x) = int_{X} q(x | v) p(v) dv
       = lim_{Q->infty} sum{ prob[i] q(x | loc=sum_k^K lambda[k;i] loc[k],
                                            scale=sum_k^K lambda[k;i] scale[k])
                            : i=0, ..., Q-1 }
  ```

  where `q(x | v)` is a vector version of the `distribution` argument and `p(v)`
  is a SoftmaxNormal parameterized by `mix_loc` and `mix_scale`. The
  vector-ization of `distribution` entails an affine transformation of iid
  samples from `distribution`.  The `prob` term is from quadrature and
  `lambda[k] = sigmoid(mix_loc[k] + sqrt(2) mix_scale[k] grid[k])` where the
  `grid` points correspond to the `prob`s.

  In the non-approximation case, a draw from the mixture distribution (the
  "prior") represents the convex weights for different affine transformations.
  I.e., draw a mixing vector `v` (from the `K-1`-simplex) and let the final
  sample be: `y = (sum_k^K v[k] scale[k]) @ x + (sum_k^K v[k] loc[k])` where `@`
  denotes matrix multiplication.  However, the non-approximate distribution does
  not have an analytical probability density function (pdf). Therefore the
  `VectorDiffeomixture` class implements an approximation based on
  [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  [Gauss--Hermite quadrature](
  https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)). I.e., in
  Note: although the `VectorDiffeomixture` is approximately the
  `SoftmaxNormal-Distribution` compound distribution, it is itself a valid
  distribution. It possesses a `sample`, `log_prob`, `mean`, `covariance` which
  are all mutually consistent.

  #### Intended Use

  This distribution is noteworthy because it implements a mixture of
  `Vector`-ized distributions yet has samples differentiable in the
  distribution's parameters (aka "reparameterized"). It has an analytical
  density function with `O(dKQ)` complexity. `d` is the vector dimensionality,
  `K` is the number of components, and `Q` is the number of quadrature points.
  These properties make it well-suited for Bayesian Variational Inference, i.e.,
  as a surrogate family for the posterior.

  For large values of `mix_scale`, the `VectorDistribution` behaves increasingly
  like a discrete mixture. (In most cases this limit is only achievable by also
  increasing the quadrature polynomial degree, `Q`.)

  The term `Vector` is consistent with similar named Tensorflow `Distribution`s.
  For more details, see the "About `Vector` distributions in Tensorflow."
  section.

  The term `Diffeomixture` is a portmanteau of
  [diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism) and [compound
  mixture](https://en.wikipedia.org/wiki/Compound_probability_distribution). For
  more details, see the "About `Diffeomixture`s and reparametrization.`"
  section.

  #### Mathematical Details

  The `VectorDiffeomixture` approximates a SoftmaxNormal-mixed ("prior")
  [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution).
  Using variable-substitution and [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  [Gauss--Hermite quadrature](
  https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)) we can
  redefine the distribution to be a parameter-less convex combination of `K`
  different affine combinations of a `d` iid samples from `distribution`.

  That is, defined over `R**d` this distribution is parameterized by a
  (batch of) length-`K` `mix_loc` and `mix_scale` vectors, a length-`K` list of
  (a batch of) length-`d` `loc` vectors, and a length-`K` list of `scale`
  `LinearOperator`s each operating on a (batch of) length-`d` vector space.
  Finally, a `distribution` parameter specifies the underlying base distribution
  which is "lifted" to become multivariate ("lifting" is the same concept as in
  `TransformedDistribution`).

  The probability density function (pdf) is,

  ```none
  pdf(y; mix_loc, mix_scale, loc, scale, phi)
    = sum{ prob[i] phi(f_inverse(x; i)) / abs(det(interp_scale[i]))
          : i=0, ..., Q-1 }
  ```

  where, `phi` is the base distribution pdf, and,

  ```none
  f_inverse(x; i) = inv(interp_scale[i]) @ (x - interp_loc[i])
  interp_loc[i]   = sum{ lambda[k; i] loc[k]   : k=0, ..., K-1 }
  interp_scale[i] = sum{ lambda[k; i] scale[k] : k=0, ..., K-1 }
  ```

  and,

  ```none
  grid, weight = np.polynomial.hermite.hermgauss(quadrature_size)
  prob[k]   = weight[k] / sqrt(pi)
  lambda[k; i] = sigmoid(mix_loc[k] + sqrt(2) mix_scale[k] grid[i])
  ```

  The distribution corresponding to `phi` must be a scalar-batch, scalar-event
  distribution. Typically it is reparameterized. If not, it must be a function
  of non-trainable parameters.

  WARNING: If you backprop through a VectorDiffeomixture sample and the "base"
  distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
  variables, then the gradient is not guaranteed correct!

  #### About `Vector` distributions in TensorFlow.

  The `VectorDiffeomixture` is a non-standard distribution that has properties
  particularly useful in [variational Bayesian
  methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).

  Conditioned on a draw from the SoftmaxNormal, `Y|v` is a vector whose
  components are linear combinations of affine transformations, thus is itself
  an affine transformation. Therefore `Y|v` lives in the vector space generated
  by vectors of affine-transformed distributions.

  Note: The marginals `Y_1|v, ..., Y_d|v` are *not* generally identical to some
  parameterization of `distribution`.  This is due to the fact that the sum of
  draws from `distribution` are not generally itself the same `distribution`.

  #### About `Diffeomixture`s and reparameterization.

  The `VectorDiffeomixture` is designed to be reparameterized, i.e., its
  parameters are only used to transform samples from a distribution which has no
  trainable parameters. This property is important because backprop stops at
  sources of stochasticity. That is, as long as the parameters are used *after*
  the underlying source of stochasticity, the computed gradient is accurate.

  Reparametrization means that we can use gradient-descent (via backprop) to
  optimize Monte-Carlo objectives. Such objectives are a finite-sample
  approximation of an expectation and arise throughout scientific computing.

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Create two batches of VectorDiffeomixtures, one with mix_loc=[0.] and
  # another with mix_loc=[1]. In both cases, `K=2` and the affine
  # transformations involve:
  # k=0: loc=zeros(dims)  scale=LinearOperatorScaledIdentity
  # k=1: loc=[2.]*dims    scale=LinOpDiag
  dims = 5
  vdm = tfd.VectorDiffeomixture(
      mix_loc=[[0.], [1]],
      mix_scale=[1.],
      distribution=tfd.Normal(loc=0., scale=1.),
      loc=[
          None,  # Equivalent to `np.zeros(dims, dtype=np.float32)`.
          np.float32([2.]*dims),
      ],
      scale=[
          tf.linalg.LinearOperatorScaledIdentity(
            num_rows=dims,
            multiplier=np.float32(1.1),
            is_positive_definite=True),
          tf.linalg.LinearOperatorDiag(
            diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
            is_positive_definite=True),
      ],
      validate_args=True)
  """

  def __init__(self,
               mix_loc,
               mix_scale,
               distribution,
               loc=None,
               scale=None,
               quadrature_size=8,
               quadrature_fn=quadrature_scheme_softmaxnormal_quantiles,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorDiffeomixture"):
    """Constructs the VectorDiffeomixture on `R**d`.

    Args:
      mix_loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`. Represents
        the `location` parameter of the SoftmaxNormal used for selecting one of
        the `K` affine transformations.
      mix_scale: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
        Represents the `scale` parameter of the SoftmaxNormal used for selecting
        one of the `K` affine transformations.
      distribution: `tf.Distribution`-like instance. Distribution from which `d`
        iid samples are used as input to the selected affine transformation.
        Must be a scalar-batch, scalar-event distribution.  Typically
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
        a function of non-trainable parameters. WARNING: If you backprop through
        a VectorDiffeomixture sample and the `distribution` is not
        `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
        the gradient will be incorrect!
      loc: Length-`K` list of `float`-type `Tensor`s. The `k`-th element
        represents the `shift` used for the `k`-th affine transformation.  If
        the `k`-th item is `None`, `loc` is implicitly `0`.  When specified,
        must have shape `[B1, ..., Bb, d]` where `b >= 0` and `d` is the event
        size.
      scale: Length-`K` list of `LinearOperator`s. Each should be
        positive-definite and operate on a `d`-dimensional vector space. The
        `k`-th element represents the `scale` used for the `k`-th affine
        transformation. `LinearOperator`s must have shape `[B1, ..., Bb, d, d]`,
        `b >= 0`, i.e., characterizes `b`-batches of `d x d` matrices
      quadrature_size: Python `int` scalar representing number of
        quadrature points.
      quadrature_fn: Python callable taking `mix_loc`, `mix_scale`,
        `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
        representing the SoftmaxNormal grid and corresponding normalized weight.
        normalized) weight.
        Default value: `quadrature_scheme_softmaxnormal_quantiles`.
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
      ValueError: if `not scale or len(scale) < 2`.
      ValueError: if `len(loc) != len(scale)`
      ValueError: if `quadrature_grid_and_probs is not None` and
        `len(quadrature_grid_and_probs[0]) != len(quadrature_grid_and_probs[1])`
      ValueError: if `validate_args` and any not scale.is_positive_definite.
      TypeError: if any scale.dtype != scale[0].dtype.
      TypeError: if any loc.dtype != scale[0].dtype.
      NotImplementedError: if `len(scale) != 2`.
      ValueError: if `not distribution.is_scalar_batch`.
      ValueError: if `not distribution.is_scalar_event`.
    """
    parameters = locals()
    with ops.name_scope(name, values=[mix_loc, mix_scale]):
      if not scale or len(scale) < 2:
        raise ValueError("Must specify list (or list-like object) of scale "
                         "LinearOperators, one for each component with "
                         "num_component >= 2.")

      if loc is None:
        loc = [None]*len(scale)

      if len(loc) != len(scale):
        raise ValueError("loc/scale must be same-length lists "
                         "(or same-length list-like objects).")

      dtype = scale[0].dtype.base_dtype

      loc = [ops.convert_to_tensor(loc_, dtype=dtype, name="loc{}".format(k))
             if loc_ is not None else None
             for k, loc_ in enumerate(loc)]

      for k, scale_ in enumerate(scale):
        if validate_args and not scale_.is_positive_definite:
          raise ValueError("scale[{}].is_positive_definite = {} != True".format(
              k, scale_.is_positive_definite))
        if scale_.dtype.base_dtype != dtype:
          raise TypeError(
              "dtype mismatch; scale[{}].base_dtype=\"{}\" != \"{}\"".format(
                  k, scale_.dtype.base_dtype.name, dtype.name))

      self._endpoint_affine = [
          AffineLinearOperator(shift=loc_,
                               scale=scale_,
                               event_ndims=1,
                               validate_args=validate_args,
                               name="endpoint_affine_{}".format(k))
          for k, (loc_, scale_) in enumerate(zip(loc, scale))]

      # TODO(jvdillon): Remove once we support k-mixtures.
      # We make this assertion here because otherwise `grid` would need to be a
      # vector not a scalar.
      if len(scale) != 2:
        raise NotImplementedError("Currently only bimixtures are supported; "
                                  "len(scale)={} is not 2.".format(len(scale)))

      self._grid, probs = tuple(quadrature_fn(
          mix_loc, mix_scale, quadrature_size, validate_args))

      # Note: by creating the logits as `log(prob)` we ensure that
      # `self.mixture_distribution.logits` is equivalent to
      # `math_ops.log(self.mixture_distribution.probs)`.
      self._mixture_distribution = categorical_lib.Categorical(
          logits=math_ops.log(probs),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      asserts = distribution_util.maybe_check_scalar_distribution(
          distribution, dtype, validate_args)
      if asserts:
        self._grid = control_flow_ops.with_dependencies(
            asserts, self._grid)
      self._distribution = distribution

      self._interpolated_affine = [
          AffineLinearOperator(shift=loc_,
                               scale=scale_,
                               event_ndims=1,
                               validate_args=validate_args,
                               name="interpolated_affine_{}".format(k))
          for k, (loc_, scale_) in enumerate(zip(
              interpolate_loc(self._grid, loc),
              interpolate_scale(self._grid, scale)))]

      [
          self._batch_shape_,
          self._batch_shape_tensor_,
          self._event_shape_,
          self._event_shape_tensor_,
      ] = determine_batch_event_shapes(self._grid,
                                       self._endpoint_affine)

      super(VectorDiffeomixture, self).__init__(
          dtype=dtype,
          # We hard-code `FULLY_REPARAMETERIZED` because when
          # `validate_args=True` we verify that indeed
          # `distribution.reparameterization_type == FULLY_REPARAMETERIZED`. A
          # distribution which is a function of only non-trainable parameters
          # also implies we can use `FULLY_REPARAMETERIZED`. However, we cannot
          # easily test for that possibility thus we use `validate_args=False`
          # as a "back-door" to allow users a way to use non
          # `FULLY_REPARAMETERIZED` distribution. In such cases IT IS THE USERS
          # RESPONSIBILITY to verify that the base distribution is a function of
          # non-trainable parameters.
          reparameterization_type=distribution_lib.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=(
              distribution._graph_parents  # pylint: disable=protected-access
              + [loc_ for loc_ in loc if loc_ is not None]
              + [p for scale_ in scale for p in scale_.graph_parents]),
          name=name)

  @property
  def mixture_distribution(self):
    """Distribution used to select a convex combination of affine transforms."""
    return self._mixture_distribution

  @property
  def distribution(self):
    """Base scalar-event, scalar-batch distribution."""
    return self._distribution

  @property
  def grid(self):
    """Grid of mixing probabilities, one for each grid point."""
    return self._grid

  @property
  def endpoint_affine(self):
    """Affine transformation for each of `K` components."""
    return self._endpoint_affine

  @property
  def interpolated_affine(self):
    """Affine transformation for each convex combination of `K` components."""
    return self._interpolated_affine

  def _batch_shape_tensor(self):
    return self._batch_shape_tensor_

  def _batch_shape(self):
    return self._batch_shape_

  def _event_shape_tensor(self):
    return self._event_shape_tensor_

  def _event_shape(self):
    return self._event_shape_

  def _sample_n(self, n, seed=None):
    x = self.distribution.sample(
        sample_shape=concat_vectors(
            [n],
            self.batch_shape_tensor(),
            self.event_shape_tensor()),
        seed=seed)   # shape: [n, B, e]
    x = [aff.forward(x) for aff in self.endpoint_affine]

    # Get ids as a [n, batch_size]-shaped matrix, unless batch_shape=[] then get
    # ids as a [n]-shaped vector.
    batch_size = self.batch_shape.num_elements()
    if batch_size is None:
      batch_size = array_ops.reduce_prod(self.batch_shape_tensor())
    mix_batch_size = self.mixture_distribution.batch_shape.num_elements()
    if mix_batch_size is None:
      mix_batch_size = math_ops.reduce_prod(
          self.mixture_distribution.batch_shape_tensor())
    ids = self.mixture_distribution.sample(
        sample_shape=concat_vectors(
            [n],
            distribution_util.pick_vector(
                self.is_scalar_batch(),
                np.int32([]),
                [batch_size // mix_batch_size])),
        seed=distribution_util.gen_new_seed(
            seed, "vector_diffeomixture"))
    # We need to flatten batch dims in case mixture_distribution has its own
    # batch dims.
    ids = array_ops.reshape(ids, shape=concat_vectors(
        [n],
        distribution_util.pick_vector(
            self.is_scalar_batch(),
            np.int32([]),
            np.int32([-1]))))

    # Stride `components * quadrature_size` for `batch_size` number of times.
    stride = self.grid.shape.with_rank_at_least(
        2)[-2:].num_elements()
    if stride is None:
      stride = array_ops.reduce_prod(
          array_ops.shape(self.grid)[-2:])
    offset = math_ops.range(start=0,
                            limit=batch_size * stride,
                            delta=stride,
                            dtype=ids.dtype)

    weight = array_ops.gather(
        array_ops.reshape(self.grid, shape=[-1]),
        ids + offset)
    weight = weight[..., array_ops.newaxis]

    if len(x) != 2:
      # We actually should have already triggered this exception. However as a
      # policy we're putting this exception wherever we exploit the bimixture
      # assumption.
      raise NotImplementedError("Currently only bimixtures are supported; "
                                "len(scale)={} is not 2.".format(len(x)))

    # Alternatively:
    # x = weight * x[0] + (1. - weight) * x[1]
    x = weight * (x[0] - x[1]) + x[1]

    return x

  def _log_prob(self, x):
    # By convention, we always put the grid points right-most.
    y = array_ops.stack(
        [aff.inverse(x) for aff in self.interpolated_affine],
        axis=-1)
    log_prob = math_ops.reduce_sum(self.distribution.log_prob(y), axis=-2)
    # Because the affine transformation has a constant Jacobian, it is the case
    # that `affine.fldj(x) = -affine.ildj(x)`. This is not true in general.
    fldj = array_ops.stack(
        [aff.forward_log_det_jacobian(x) for aff in self.interpolated_affine],
        axis=-1)
    return math_ops.reduce_logsumexp(
        self.mixture_distribution.logits - fldj + log_prob, axis=-1)

  def _mean(self):
    p = self._expand_mix_distribution_probs()
    m = self._expand_base_distribution_mean()
    mean = None
    for k, aff in enumerate(self.interpolated_affine):
      # aff.forward is going to do this:
      # y = array_ops.squeeze(aff.scale.matmul(m), axis=[-1])
      # if aff.shift is not None:
      #   y += aff.shift
      mean = add(mean, p[..., k] * aff.forward(m))
    return mean

  def _covariance(self):
    # Law of total variance:
    #
    # Cov[Z] = E[Cov[Z | V]] + Cov[E[Z | V]]
    #
    # where,
    #
    # E[Cov[Z | V]] = sum_i mix_prob[i] Scale[i]
    # Cov[E[Z | V]] = sum_i mix_prob[i] osquare(loc[i])
    #                  - osquare(sum_i mix_prob[i] loc[i])
    #
    # osquare(x) = x.transpose @ x
    return add(
        self._mean_of_covariance_given_quadrature_component(diag_only=False),
        self._covariance_of_mean_given_quadrature_component(diag_only=False))

  def _variance(self):
    # Equivalent to: tf.diag_part(self._covariance()),
    return add(
        self._mean_of_covariance_given_quadrature_component(diag_only=True),
        self._covariance_of_mean_given_quadrature_component(diag_only=True))

  def _mean_of_covariance_given_quadrature_component(self, diag_only):
    p = self.mixture_distribution.probs

    # To compute E[Cov(Z|V)], we'll add matrices within three categories:
    # scaled-identity, diagonal, and full. Then we'll combine these at the end.
    scaled_identity = None
    diag = None
    full = None

    for k, aff in enumerate(self.interpolated_affine):
      s = aff.scale  # Just in case aff.scale has side-effects, we'll call once.
      if (s is None
          or isinstance(s, linop_identity_lib.LinearOperatorIdentity)):
        scaled_identity = add(scaled_identity, p[..., k, array_ops.newaxis])
      elif isinstance(s, linop_identity_lib.LinearOperatorScaledIdentity):
        scaled_identity = add(scaled_identity, (p[..., k, array_ops.newaxis] *
                                                math_ops.square(s.multiplier)))
      elif isinstance(s, linop_diag_lib.LinearOperatorDiag):
        diag = add(diag, (p[..., k, array_ops.newaxis] *
                          math_ops.square(s.diag_part())))
      else:
        x = (p[..., k, array_ops.newaxis, array_ops.newaxis] *
             s.matmul(s.to_dense(), adjoint_arg=True))
        if diag_only:
          x = array_ops.matrix_diag_part(x)
        full = add(full, x)

    # We must now account for the fact that the base distribution might have a
    # non-unity variance. Recall that `Cov(SX+m) = S.T Cov(X) S = S.T S Var(X)`.
    # We can scale by `Var(X)` (vs `Cov(X)`) since X corresponds to `d` iid
    # samples from a scalar-event distribution.
    v = self.distribution.variance()
    if scaled_identity is not None:
      scaled_identity *= v
    if diag is not None:
      diag *= v[..., array_ops.newaxis]
    if full is not None:
      full *= v[..., array_ops.newaxis]

    if diag_only:
      # Apparently we don't need the full matrix, just the diagonal.
      r = add(diag, full)
      if r is None and scaled_identity is not None:
        ones = array_ops.ones(self.event_shape_tensor(), dtype=self.dtype)
        return scaled_identity * ones
      return add(r, scaled_identity)

    # `None` indicates we don't know if the result is positive-definite.
    is_positive_definite = (True if all(aff.scale.is_positive_definite
                                        for aff in self.endpoint_affine)
                            else None)

    to_add = []
    if diag is not None:
      to_add.append(linop_diag_lib.LinearOperatorDiag(
          diag=diag,
          is_positive_definite=is_positive_definite))
    if full is not None:
      to_add.append(linop_full_lib.LinearOperatorFullMatrix(
          matrix=full,
          is_positive_definite=is_positive_definite))
    if scaled_identity is not None:
      to_add.append(linop_identity_lib.LinearOperatorScaledIdentity(
          num_rows=self.event_shape_tensor()[0],
          multiplier=scaled_identity,
          is_positive_definite=is_positive_definite))

    return (linop_add_lib.add_operators(to_add)[0].to_dense()
            if to_add else None)

  def _covariance_of_mean_given_quadrature_component(self, diag_only):
    square = math_ops.square if diag_only else vec_osquare

    p = self._expand_mix_distribution_probs()
    if not diag_only:
      p = p[..., array_ops.newaxis, :]  # Assuming event.ndims=1.
    m = self._expand_base_distribution_mean()

    cov_e_z_given_v = None
    e_z_given_v = self._mean()
    for k, aff in enumerate(self.interpolated_affine):
      y = aff.forward(m)
      cov_e_z_given_v = add(cov_e_z_given_v,
                            p[..., k] * square(y - e_z_given_v))

    return cov_e_z_given_v

  def _expand_base_distribution_mean(self):
    """Ensures `self.distribution.mean()` has `[batch, event]` shape."""
    single_draw_shape = concat_vectors(self.batch_shape_tensor(),
                                       self.event_shape_tensor())
    m = array_ops.reshape(
        self.distribution.mean(),  # A scalar.
        shape=array_ops.ones_like(single_draw_shape,
                                  dtype=dtypes.int32))
    m = array_ops.tile(m, multiples=single_draw_shape)
    m.set_shape(self.batch_shape.concatenate(self.event_shape))
    return m

  def _expand_mix_distribution_probs(self):
    p = self.mixture_distribution.probs  # [B, deg]
    deg = p.shape.with_rank_at_least(1)[-1].value
    if deg is None:
      deg = array_ops.shape(p)[-1]
    event_ndims = self.event_shape.ndims
    if event_ndims is None:
      event_ndims = array_ops.shape(self.event_shape_tensor())[0]
    expand_shape = array_ops.concat([
        self.mixture_distribution.batch_shape_tensor(),
        array_ops.ones([event_ndims], dtype=dtypes.int32),
        [deg],
    ], axis=0)
    return array_ops.reshape(p, shape=expand_shape)


def maybe_check_quadrature_param(param, name, validate_args):
  """Helper which checks validity of `loc` and `scale` init args."""
  with ops.name_scope(name="check_" + name, values=[param]):
    assertions = []
    if param.shape.ndims is not None:
      if param.shape.ndims == 0:
        raise ValueError("Mixing params must be a (batch of) vector; "
                         "{}.rank={} is not at least one.".format(
                             name, param.shape.ndims))
    elif validate_args:
      assertions.append(check_ops.assert_rank_at_least(
          param, 1,
          message=("Mixing params must be a (batch of) vector; "
                   "{}.rank is not at least one.".format(
                       name))))

    # TODO(jvdillon): Remove once we support k-mixtures.
    if param.shape.with_rank_at_least(1)[-1] is not None:
      if param.shape[-1].value != 1:
        raise NotImplementedError("Currently only bimixtures are supported; "
                                  "{}.shape[-1]={} is not 1.".format(
                                      name, param.shape[-1].value))
    elif validate_args:
      assertions.append(check_ops.assert_equal(
          array_ops.shape(param)[-1], 1,
          message=("Currently only bimixtures are supported; "
                   "{}.shape[-1] is not 1.".format(name))))

    if assertions:
      return control_flow_ops.with_dependencies(assertions, param)
    return param


def determine_batch_event_shapes(grid, endpoint_affine):
  """Helper to infer batch_shape and event_shape."""
  with ops.name_scope(name="determine_batch_event_shapes"):
    # grid  # shape: [B, k, q]
    # endpoint_affine     # len=k, shape: [B, d, d]
    batch_shape = grid.shape[:-2]
    batch_shape_tensor = array_ops.shape(grid)[:-2]
    event_shape = None
    event_shape_tensor = None

    def _set_event_shape(shape, shape_tensor):
      if event_shape is None:
        return shape, shape_tensor
      return (array_ops.broadcast_static_shape(event_shape, shape),
              array_ops.broadcast_dynamic_shape(
                  event_shape_tensor, shape_tensor))

    for aff in endpoint_affine:
      if aff.shift is not None:
        batch_shape = array_ops.broadcast_static_shape(
            batch_shape, aff.shift.shape[:-1])
        batch_shape_tensor = array_ops.broadcast_dynamic_shape(
            batch_shape_tensor, array_ops.shape(aff.shift)[:-1])
        event_shape, event_shape_tensor = _set_event_shape(
            aff.shift.shape[-1:], array_ops.shape(aff.shift)[-1:])

      if aff.scale is not None:
        batch_shape = array_ops.broadcast_static_shape(
            batch_shape, aff.scale.batch_shape)
        batch_shape_tensor = array_ops.broadcast_dynamic_shape(
            batch_shape_tensor, aff.scale.batch_shape_tensor())
        event_shape, event_shape_tensor = _set_event_shape(
            tensor_shape.TensorShape([aff.scale.range_dimension]),
            aff.scale.range_dimension_tensor()[array_ops.newaxis])

    return batch_shape, batch_shape_tensor, event_shape, event_shape_tensor


def interpolate_loc(grid, loc):
  """Helper which interpolates between two locs."""
  if len(loc) != 2:
    raise NotImplementedError("Currently only bimixtures are supported; "
                              "len(scale)={} is not 2.".format(len(loc)))
  deg = grid.shape.with_rank_at_least(1)[-1].value
  if deg is None:
    raise ValueError("Num quadrature grid points must be known prior "
                     "to graph execution.")
  with ops.name_scope("interpolate_loc", values=[grid, loc]):
    if loc is None or loc[0] is None and loc[1] is None:
      return [None]*deg
    # shape: [B, 1, k, deg]
    w = grid[..., array_ops.newaxis, :, :]
    loc = [x[..., array_ops.newaxis]                   # shape: [B, e, 1]
           if x is not None else None for x in loc]
    if loc[0] is None:
      x = w[..., 1, :] * loc[1]                        # shape: [B, e, deg]
    elif loc[1] is None:
      x = w[..., 0, :] * loc[0]                        # shape: [B, e, deg]
    else:
      delta = loc[0] - loc[1]
      x = w[..., 0, :] * delta + loc[1]                # shape: [B, e, deg]
    return [x[..., k] for k in range(deg)]             # list(shape:[B, e])


def interpolate_scale(grid, scale):
  """Helper which interpolates between two scales."""
  if len(scale) != 2:
    raise NotImplementedError("Currently only bimixtures are supported; "
                              "len(scale)={} is not 2.".format(len(scale)))
  deg = grid.shape.with_rank_at_least(1)[-1].value
  if deg is None:
    raise ValueError("Num quadrature grid points must be known prior "
                     "to graph execution.")
  with ops.name_scope("interpolate_scale", values=[grid]):
    return [linop_add_lib.add_operators([
        linop_scale(grid[..., k, q], s)
        for k, s in enumerate(scale)
    ])[0] for q in range(deg)]


def linop_scale(w, op):
  # We assume w > 0. (This assumption only relates to the is_* attributes.)
  with ops.name_scope("linop_scale", values=[w]):
    # TODO(b/35301104): LinearOperatorComposition doesn't combine operators, so
    # special case combinations here. Once it does, this function can be
    # replaced by:
    #     return linop_composition_lib.LinearOperatorComposition([
    #         scaled_identity(w), op])
    def scaled_identity(w):
      return linop_identity_lib.LinearOperatorScaledIdentity(
          num_rows=op.range_dimension_tensor(),
          multiplier=w,
          is_non_singular=op.is_non_singular,
          is_self_adjoint=op.is_self_adjoint,
          is_positive_definite=op.is_positive_definite)
    if isinstance(op, linop_identity_lib.LinearOperatorIdentity):
      return scaled_identity(w)
    if isinstance(op, linop_identity_lib.LinearOperatorScaledIdentity):
      return scaled_identity(w * op.multiplier)
    if isinstance(op, linop_diag_lib.LinearOperatorDiag):
      return linop_diag_lib.LinearOperatorDiag(
          diag=w[..., array_ops.newaxis] * op.diag_part(),
          is_non_singular=op.is_non_singular,
          is_self_adjoint=op.is_self_adjoint,
          is_positive_definite=op.is_positive_definite)
    if isinstance(op, linop_tril_lib.LinearOperatorLowerTriangular):
      return linop_tril_lib.LinearOperatorLowerTriangular(
          tril=w[..., array_ops.newaxis, array_ops.newaxis] * op.to_dense(),
          is_non_singular=op.is_non_singular,
          is_self_adjoint=op.is_self_adjoint,
          is_positive_definite=op.is_positive_definite)
    raise NotImplementedError(
        "Unsupported Linop type ({})".format(type(op).__name__))


def concat_vectors(*args):
  """Concatenates input vectors, statically if possible."""
  args_ = [distribution_util.static_value(x) for x in args]
  if any(vec is None for vec in args_):
    return array_ops.concat(args, axis=0)
  return [val for vec in args_ for val in vec]


def add(x, y):
  """Adds inputs; interprets `None` as zero."""
  if x is None:
    return y
  if y is None:
    return x
  return x + y


def vec_osquare(x):
  """Computes the outer-product of a (batch of) vector, i.e., x.T x."""
  return x[..., :, array_ops.newaxis] * x[..., array_ops.newaxis, :]


def softmax(x, axis, name=None):
  """Equivalent to tf.nn.softmax but works around b/70297725."""
  with ops.name_scope(name, "softmax", [x, axis]):
    x = ops.convert_to_tensor(x, name="x")
    ndims = (x.shape.ndims if x.shape.ndims is not None
             else array_ops.rank(x, name="ndims"))
    axis = ops.convert_to_tensor(axis, dtype=dtypes.int32, name="axis")
    axis_ = tensor_util.constant_value(axis)
    if axis_ is not None:
      axis = np.int(ndims + axis_ if axis_ < 0 else axis_)
    else:
      axis = array_ops.where(axis < 0, ndims + axis, axis)
  return nn_ops.softmax(x, axis=axis)
