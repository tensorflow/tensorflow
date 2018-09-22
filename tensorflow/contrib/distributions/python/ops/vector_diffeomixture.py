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
from tensorflow.python.ops.linalg import linear_operator_addition as linop_add_lib
from tensorflow.python.ops.linalg import linear_operator_diag as linop_diag_lib
from tensorflow.python.ops.linalg import linear_operator_full_matrix as linop_full_lib
from tensorflow.python.ops.linalg import linear_operator_identity as linop_identity_lib
from tensorflow.python.ops.linalg import linear_operator_lower_triangular as linop_tril_lib
from tensorflow.python.util import deprecation


__all__ = [
    "VectorDiffeomixture",
    "quadrature_scheme_softmaxnormal_gauss_hermite",
    "quadrature_scheme_softmaxnormal_quantiles",
]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def quadrature_scheme_softmaxnormal_gauss_hermite(
    normal_loc, normal_scale, quadrature_size,
    validate_args=False, name=None):
  """Use Gauss-Hermite quadrature to form quadrature on `K - 1` simplex.

  A `SoftmaxNormal` random variable `Y` may be generated via

  ```
  Y = SoftmaxCentered(X),
  X = Normal(normal_loc, normal_scale)
  ```

  Note: for a given `quadrature_size`, this method is generally less accurate
  than `quadrature_scheme_softmaxnormal_quantiles`.

  Args:
    normal_loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      The location parameter of the Normal used to construct the SoftmaxNormal.
    normal_scale: `float`-like `Tensor`. Broadcastable with `normal_loc`.
      The scale parameter of the Normal used to construct the SoftmaxNormal.
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
                      [normal_loc, normal_scale]):
    normal_loc = ops.convert_to_tensor(normal_loc, name="normal_loc")
    dt = normal_loc.dtype.base_dtype
    normal_scale = ops.convert_to_tensor(
        normal_scale, dtype=dt, name="normal_scale")

    normal_scale = maybe_check_quadrature_param(
        normal_scale, "normal_scale", validate_args)

    grid, probs = np.polynomial.hermite.hermgauss(deg=quadrature_size)
    grid = grid.astype(dt.dtype.as_numpy_dtype)
    probs = probs.astype(dt.dtype.as_numpy_dtype)
    probs /= np.linalg.norm(probs, ord=1, keepdims=True)
    probs = ops.convert_to_tensor(probs, name="probs", dtype=dt)

    grid = softmax(
        -distribution_util.pad(
            (normal_loc[..., array_ops.newaxis] +
             np.sqrt(2.) * normal_scale[..., array_ops.newaxis] * grid),
            axis=-2,
            front=True),
        axis=-2)  # shape: [B, components, deg]

    return grid, probs


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def quadrature_scheme_softmaxnormal_quantiles(
    normal_loc, normal_scale, quadrature_size,
    validate_args=False, name=None):
  """Use SoftmaxNormal quantiles to form quadrature on `K - 1` simplex.

  A `SoftmaxNormal` random variable `Y` may be generated via

  ```
  Y = SoftmaxCentered(X),
  X = Normal(normal_loc, normal_scale)
  ```

  Args:
    normal_loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`, B>=0.
      The location parameter of the Normal used to construct the SoftmaxNormal.
    normal_scale: `float`-like `Tensor`. Broadcastable with `normal_loc`.
      The scale parameter of the Normal used to construct the SoftmaxNormal.
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
  with ops.name_scope(name, "softmax_normal_grid_and_probs",
                      [normal_loc, normal_scale]):
    normal_loc = ops.convert_to_tensor(normal_loc, name="normal_loc")
    dt = normal_loc.dtype.base_dtype
    normal_scale = ops.convert_to_tensor(
        normal_scale, dtype=dt, name="normal_scale")

    normal_scale = maybe_check_quadrature_param(
        normal_scale, "normal_scale", validate_args)

    dist = normal_lib.Normal(loc=normal_loc, scale=normal_scale)

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
      quantiles = SoftmaxCentered().forward(quantiles)
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

  A vector diffeomixture (VDM) is a distribution parameterized by a convex
  combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
  `scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
  [compound distribution]
  (https://en.wikipedia.org/wiki/Compound_probability_distribution)

  ```none
  p(x) = int p(x | z) p(z) dz,
  where z is in the K-simplex, and
  p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
  ```

  The integral `int p(x | z) p(z) dz` is approximated with a quadrature scheme
  adapted to the mixture density `p(z)`.  The `N` quadrature points `z_{N, n}`
  and weights `w_{N, n}` (which are non-negative and sum to 1) are chosen
  such that

  ```q_N(x) := sum_{n=1}^N w_{n, N} p(x | z_{N, n}) --> p(x)```

  as `N --> infinity`.

  Since `q_N(x)` is in fact a mixture (of `N` points), we may sample from
  `q_N` exactly.  It is important to note that the VDM is *defined* as `q_N`
  above, and *not* `p(x)`.  Therefore, sampling and pdf may be implemented as
  exact (up to floating point error) methods.

  A common choice for the conditional `p(x | z)` is a multivariate Normal.

  The implemented marginal `p(z)` is the `SoftmaxNormal`, which is a
  `K-1` dimensional Normal transformed by a `SoftmaxCentered` bijector, making
  it a density on the `K`-simplex.  That is,

  ```
  Z = SoftmaxCentered(X),
  X = Normal(mix_loc / temperature, 1 / temperature)
  ```

  The default quadrature scheme chooses `z_{N, n}` as `N` midpoints of
  the quantiles of `p(z)` (generalized quantiles if `K > 2`).

  See [Dillon and Langmore (2018)][1] for more details.

  #### About `Vector` distributions in TensorFlow.

  The `VectorDiffeomixture` is a non-standard distribution that has properties
  particularly useful in [variational Bayesian
  methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).

  Conditioned on a draw from the SoftmaxNormal, `X|z` is a vector whose
  components are linear combinations of affine transformations, thus is itself
  an affine transformation.

  Note: The marginals `X_1|v, ..., X_d|v` are *not* generally identical to some
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

  WARNING: If you backprop through a VectorDiffeomixture sample and the "base"
  distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
  variables, then the gradient is not guaranteed correct!

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create two batches of VectorDiffeomixtures, one with mix_loc=[0.],
  # another with mix_loc=[1]. In both cases, `K=2` and the affine
  # transformations involve:
  # k=0: loc=zeros(dims)  scale=LinearOperatorScaledIdentity
  # k=1: loc=[2.]*dims    scale=LinOpDiag
  dims = 5
  vdm = tfd.VectorDiffeomixture(
      mix_loc=[[0.], [1]],
      temperature=[1.],
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
  ```

  #### References

  [1]: Joshua Dillon and Ian Langmore. Quadrature Compound: An approximating
       family of distributions. _arXiv preprint arXiv:1801.03080_, 2018.
       https://arxiv.org/abs/1801.03080
  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               mix_loc,
               temperature,
               distribution,
               loc=None,
               scale=None,
               quadrature_size=8,
               quadrature_fn=quadrature_scheme_softmaxnormal_quantiles,
               validate_args=False,
               allow_nan_stats=True,
               name="VectorDiffeomixture"):
    """Constructs the VectorDiffeomixture on `R^d`.

    The vector diffeomixture (VDM) approximates the compound distribution

    ```none
    p(x) = int p(x | z) p(z) dz,
    where z is in the K-simplex, and
    p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
    ```

    Args:
      mix_loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
        In terms of samples, larger `mix_loc[..., k]` ==>
        `Z` is more likely to put more weight on its `kth` component.
      temperature: `float`-like `Tensor`. Broadcastable with `mix_loc`.
        In terms of samples, smaller `temperature` means one component is more
        likely to dominate.  I.e., smaller `temperature` makes the VDM look more
        like a standard mixture of `K` components.
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
        quadrature points.  Larger `quadrature_size` means `q_N(x)` better
        approximates `p(x)`.
      quadrature_fn: Python callable taking `normal_loc`, `normal_scale`,
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
    parameters = dict(locals())
    with ops.name_scope(name, values=[mix_loc, temperature]) as name:
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
                               validate_args=validate_args,
                               name="endpoint_affine_{}".format(k))
          for k, (loc_, scale_) in enumerate(zip(loc, scale))]

      # TODO(jvdillon): Remove once we support k-mixtures.
      # We make this assertion here because otherwise `grid` would need to be a
      # vector not a scalar.
      if len(scale) != 2:
        raise NotImplementedError("Currently only bimixtures are supported; "
                                  "len(scale)={} is not 2.".format(len(scale)))

      mix_loc = ops.convert_to_tensor(
          mix_loc, dtype=dtype, name="mix_loc")
      temperature = ops.convert_to_tensor(
          temperature, dtype=dtype, name="temperature")
      self._grid, probs = tuple(quadrature_fn(
          mix_loc / temperature,
          1. / temperature,
          quadrature_size,
          validate_args))

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
    # At this point, weight flattened all batch dims into one.
    # We also need to append a singleton to broadcast with event dims.
    if self.batch_shape.is_fully_defined():
      new_shape = [-1] + self.batch_shape.as_list() + [1]
    else:
      new_shape = array_ops.concat(
          ([-1], self.batch_shape_tensor(), [1]), axis=0)
    weight = array_ops.reshape(weight, shape=new_shape)

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
    fldj = array_ops.stack([
        aff.forward_log_det_jacobian(
            x,
            event_ndims=array_ops.rank(self.event_shape_tensor())
        ) for aff in self.interpolated_affine], axis=-1)
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
    scale_identity_multiplier = None
    diag = None
    full = None

    for k, aff in enumerate(self.interpolated_affine):
      s = aff.scale  # Just in case aff.scale has side-effects, we'll call once.
      if (s is None
          or isinstance(s, linop_identity_lib.LinearOperatorIdentity)):
        scale_identity_multiplier = add(scale_identity_multiplier,
                                        p[..., k, array_ops.newaxis])
      elif isinstance(s, linop_identity_lib.LinearOperatorScaledIdentity):
        scale_identity_multiplier = add(
            scale_identity_multiplier,
            (p[..., k, array_ops.newaxis] * math_ops.square(s.multiplier)))
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
    # non-unity variance. Recall that, since X ~ iid Law(X_0),
    #   `Cov(SX+m) = S Cov(X) S.T = S S.T Diag(Var(X_0))`.
    # We can scale by `Var(X)` (vs `Cov(X)`) since X corresponds to `d` iid
    # samples from a scalar-event distribution.
    v = self.distribution.variance()
    if scale_identity_multiplier is not None:
      scale_identity_multiplier *= v
    if diag is not None:
      diag *= v[..., array_ops.newaxis]
    if full is not None:
      full *= v[..., array_ops.newaxis]

    if diag_only:
      # Apparently we don't need the full matrix, just the diagonal.
      r = add(diag, full)
      if r is None and scale_identity_multiplier is not None:
        ones = array_ops.ones(self.event_shape_tensor(), dtype=self.dtype)
        return scale_identity_multiplier[..., array_ops.newaxis] * ones
      return add(r, scale_identity_multiplier)

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
    if scale_identity_multiplier is not None:
      to_add.append(linop_identity_lib.LinearOperatorScaledIdentity(
          num_rows=self.event_shape_tensor()[0],
          multiplier=scale_identity_multiplier,
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def concat_vectors(*args):
  """Concatenates input vectors, statically if possible."""
  args_ = [distribution_util.static_value(x) for x in args]
  if any(vec is None for vec in args_):
    return array_ops.concat(args, axis=0)
  return [val for vec in args_ for val in vec]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def add(x, y):
  """Adds inputs; interprets `None` as zero."""
  if x is None:
    return y
  if y is None:
    return x
  return x + y


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def vec_osquare(x):
  """Computes the outer-product of a (batch of) vector, i.e., x.T x."""
  return x[..., :, array_ops.newaxis] * x[..., array_ops.newaxis, :]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
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
