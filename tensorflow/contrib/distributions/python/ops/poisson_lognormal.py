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
"""The PoissonLogNormalQuadratureCompound distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import poisson as poisson_lib
from tensorflow.contrib.distributions.python.ops.bijectors.exp import Exp
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import categorical as categorical_lib
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import transformed_distribution as transformed_lib


__all__ = [
    "PoissonLogNormalQuadratureCompound",
    "quadrature_scheme_lognormal_gauss_hermite",
    "quadrature_scheme_lognormal_quantiles",
]


def quadrature_scheme_lognormal_gauss_hermite(
    loc, scale, quadrature_size,
    validate_args=False, name=None):  # pylint: disable=unused-argument
  """Use Gauss-Hermite quadrature to form quadrature on positive-reals.

  Note: for a given `quadrature_size`, this method is generally less accurate
  than `quadrature_scheme_lognormal_quantiles`.

  Args:
    loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
      the LogNormal prior.
    scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
      the LogNormal prior.
    quadrature_size: Python `int` scalar representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: (Batch of) length-`quadrature_size` vectors representing the
      `log_rate` parameters of a `Poisson`.
    probs: (Batch of) length-`quadrature_size` vectors representing the
      weight associate with each `grid` value.
  """
  with ops.name_scope(name, "vector_diffeomixture_quadrature_gauss_hermite",
                      [loc, scale]):
    grid, probs = np.polynomial.hermite.hermgauss(deg=quadrature_size)
    grid = grid.astype(loc.dtype.as_numpy_dtype)
    probs = probs.astype(loc.dtype.as_numpy_dtype)
    probs /= np.linalg.norm(probs, ord=1, keepdims=True)
    probs = ops.convert_to_tensor(probs, name="probs", dtype=loc.dtype)
    # The following maps the broadcast of `loc` and `scale` to each grid
    # point, i.e., we are creating several log-rates that correspond to the
    # different Gauss-Hermite quadrature points and (possible) batches of
    # `loc` and `scale`.
    grid = (loc[..., array_ops.newaxis]
            + np.sqrt(2.) * scale[..., array_ops.newaxis] * grid)
    return grid, probs


def quadrature_scheme_lognormal_quantiles(
    loc, scale, quadrature_size,
    validate_args=False, name=None):
  """Use LogNormal quantiles to form quadrature on positive-reals.

  Args:
    loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
      the LogNormal prior.
    scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
      the LogNormal prior.
    quadrature_size: Python `int` scalar representing the number of quadrature
      points.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    grid: (Batch of) length-`quadrature_size` vectors representing the
      `log_rate` parameters of a `Poisson`.
    probs: (Batch of) length-`quadrature_size` vectors representing the
      weight associate with each `grid` value.
  """
  with ops.name_scope(name, "quadrature_scheme_lognormal_quantiles",
                      [loc, scale]):
    # Create a LogNormal distribution.
    dist = transformed_lib.TransformedDistribution(
        distribution=normal_lib.Normal(loc=loc, scale=scale),
        bijector=Exp(),
        validate_args=validate_args)
    batch_ndims = dist.batch_shape.ndims
    if batch_ndims is None:
      batch_ndims = array_ops.shape(dist.batch_shape_tensor())[0]

    def _compute_quantiles():
      """Helper to build quantiles."""
      # Omit {0, 1} since they might lead to Inf/NaN.
      zero = array_ops.zeros([], dtype=dist.dtype)
      edges = math_ops.linspace(zero, 1., quadrature_size + 3)[1:-1]
      # Expand edges so its broadcast across batch dims.
      edges = array_ops.reshape(edges, shape=array_ops.concat([
          [-1], array_ops.ones([batch_ndims], dtype=dtypes.int32)], axis=0))
      quantiles = dist.quantile(edges)
      # Cyclically permute left by one.
      perm = array_ops.concat([
          math_ops.range(1, 1 + batch_ndims), [0]], axis=0)
      quantiles = array_ops.transpose(quantiles, perm)
      return quantiles
    quantiles = _compute_quantiles()

    # Compute grid as quantile midpoints.
    grid = (quantiles[..., :-1] + quantiles[..., 1:]) / 2.
    # Set shape hints.
    grid.set_shape(dist.batch_shape.concatenate([quadrature_size]))

    # By construction probs is constant, i.e., `1 / quadrature_size`. This is
    # important, because non-constant probs leads to non-reparameterizable
    # samples.
    probs = array_ops.fill(
        dims=[quadrature_size],
        value=1. / math_ops.cast(quadrature_size, dist.dtype))

    return grid, probs


class PoissonLogNormalQuadratureCompound(distribution_lib.Distribution):
  """`PoissonLogNormalQuadratureCompound` distribution.

  The `PoissonLogNormalQuadratureCompound` is an approximation to a
  Poisson-LogNormal [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e.,

  ```none
  p(k|loc, scale)
  = int_{R_+} dl LogNormal(l | loc, scale) Poisson(k | l)
  approx= sum{ prob[d] Poisson(k | lambda(grid[d])) : d=0, ..., deg-1 }
  ```

  By default, the `grid` is chosen as quantiles of the `LogNormal` distribution
  parameterized by `loc`, `scale` and the `prob` vector is
  `[1. / quadrature_size]*quadrature_size`.

  In the non-approximation case, a draw from the LogNormal prior represents the
  Poisson rate parameter. Unfortunately, the non-approximate distribution lacks
  an analytical probability density function (pdf). Therefore the
  `PoissonLogNormalQuadratureCompound` class implements an approximation based
  on [quadrature](https://en.wikipedia.org/wiki/Numerical_integration).

  Note: although the `PoissonLogNormalQuadratureCompound` is approximately the
  Poisson-LogNormal compound distribution, it is itself a valid distribution.
  Viz., it possesses a `sample`, `log_prob`, `mean`, `variance`, etc. which are
  all mutually consistent.

  #### Mathematical Details

  The `PoissonLogNormalQuadratureCompound` approximates a Poisson-LogNormal
  [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution). Using
  variable-substitution and [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  based on `LogNormal` quantiles) we can redefine the distribution to be a
  parameter-less convex combination of `deg` different Poisson samples.

  That is, defined over positive integers, this distribution is parameterized
  by a (batch of) `loc` and `scale` scalars.

  The probability density function (pdf) is,

  ```none
  pdf(k | loc, scale, deg)
    = sum{ prob[d] Poisson(k | lambda=exp(grid[d]))
          : d=0, ..., deg-1 }
  ```

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Create two batches of PoissonLogNormalQuadratureCompounds, one with
  # prior `loc = 0.` and another with `loc = 1.` In both cases `scale = 1.`
  pln = tfd.PoissonLogNormalQuadratureCompound(
      loc=[0., -0.5],
      scale=1.,
      quadrature_size=10,
      validate_args=True)
  """

  def __init__(self,
               loc,
               scale,
               quadrature_size=8,
               quadrature_fn=quadrature_scheme_lognormal_quantiles,
               validate_args=False,
               allow_nan_stats=True,
               name="PoissonLogNormalQuadratureCompound"):
    """Constructs the PoissonLogNormalQuadratureCompound`.

    Note: `probs` returned by (optional) `quadrature_fn` are presumed to be
    either a length-`quadrature_size` vector or a batch of vectors in 1-to-1
    correspondence with the returned `grid`. (I.e., broadcasting is only
    partially supported.)

    Args:
      loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
        the LogNormal prior.
      scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
        the LogNormal prior.
      quadrature_size: Python `int` scalar representing the number of quadrature
        points.
      quadrature_fn: Python callable taking `loc`, `scale`,
        `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
        representing the LogNormal grid and corresponding normalized weight.
        normalized) weight.
        Default value: `quadrature_scheme_lognormal_quantiles`.
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
      TypeError: if `quadrature_grid` and `quadrature_probs` have different base
        `dtype`.
    """
    parameters = distribution_util.parent_frame_arguments()
    with ops.name_scope(name, values=[loc, scale]) as name:
      if loc is not None:
        loc = ops.convert_to_tensor(loc, name="loc")
      if scale is not None:
        scale = ops.convert_to_tensor(
            scale, dtype=None if loc is None else loc.dtype, name="scale")
      self._quadrature_grid, self._quadrature_probs = tuple(quadrature_fn(
          loc, scale, quadrature_size, validate_args))

      dt = self._quadrature_grid.dtype
      if dt.base_dtype != self._quadrature_probs.dtype.base_dtype:
        raise TypeError("Quadrature grid dtype ({}) does not match quadrature "
                        "probs dtype ({}).".format(
                            dt.name, self._quadrature_probs.dtype.name))

      self._distribution = poisson_lib.Poisson(
          log_rate=self._quadrature_grid,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      self._mixture_distribution = categorical_lib.Categorical(
          logits=math_ops.log(self._quadrature_probs),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      self._loc = loc
      self._scale = scale
      self._quadrature_size = quadrature_size

      super(PoissonLogNormalQuadratureCompound, self).__init__(
          dtype=dt,
          reparameterization_type=distribution_lib.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[loc, scale],
          name=name)

  @property
  def mixture_distribution(self):
    """Distribution which randomly selects a Poisson with quadrature param."""
    return self._mixture_distribution

  @property
  def distribution(self):
    """Base Poisson parameterized by a quadrature grid."""
    return self._distribution

  @property
  def loc(self):
    """Location parameter of the LogNormal prior."""
    return self._loc

  @property
  def scale(self):
    """Scale parameter of the LogNormal prior."""
    return self._scale

  @property
  def quadrature_size(self):
    return self._quadrature_size

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        self.distribution.batch_shape_tensor(),
        array_ops.shape(self.mixture_distribution.logits))[:-1]

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.distribution.batch_shape,
        self.mixture_distribution.logits.shape)[:-1]

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Get ids as a [n, batch_size]-shaped matrix, unless batch_shape=[] then get
    # ids as a [n]-shaped vector.
    batch_size = self.batch_shape.num_elements()
    if batch_size is None:
      batch_size = math_ops.reduce_prod(self.batch_shape_tensor())
    # We need to "sample extra" from the mixture distribution if it doesn't
    # already specify a probs vector for each batch coordinate.
    # We only support this kind of reduced broadcasting, i.e., there is exactly
    # one probs vector for all batch dims or one for each.
    ids = self._mixture_distribution.sample(
        sample_shape=concat_vectors(
            [n],
            distribution_util.pick_vector(
                self.mixture_distribution.is_scalar_batch(),
                [batch_size],
                np.int32([]))),
        seed=distribution_util.gen_new_seed(
            seed, "poisson_lognormal_quadrature_compound"))
    # We need to flatten batch dims in case mixture_distribution has its own
    # batch dims.
    ids = array_ops.reshape(ids, shape=concat_vectors(
        [n],
        distribution_util.pick_vector(
            self.is_scalar_batch(),
            np.int32([]),
            np.int32([-1]))))

    # Stride `quadrature_size` for `batch_size` number of times.
    offset = math_ops.range(start=0,
                            limit=batch_size * self._quadrature_size,
                            delta=self._quadrature_size,
                            dtype=ids.dtype)
    ids += offset
    rate = array_ops.gather(
        array_ops.reshape(self.distribution.rate, shape=[-1]), ids)
    rate = array_ops.reshape(
        rate, shape=concat_vectors([n], self.batch_shape_tensor()))
    return random_ops.random_poisson(
        lam=rate, shape=[], dtype=self.dtype, seed=seed)

  def _log_prob(self, x):
    return math_ops.reduce_logsumexp(
        (self.mixture_distribution.logits
         + self.distribution.log_prob(x[..., array_ops.newaxis])),
        axis=-1)

  def _mean(self):
    return math_ops.exp(
        math_ops.reduce_logsumexp(
            self.mixture_distribution.logits + self.distribution.log_rate,
            axis=-1))

  def _variance(self):
    return math_ops.exp(self._log_variance())

  def _stddev(self):
    return math_ops.exp(0.5 * self._log_variance())

  def _log_variance(self):
    # Following calculation is based on law of total variance:
    #
    # Var[Z] = E[Var[Z | V]] + Var[E[Z | V]]
    #
    # where,
    #
    # Z|v ~ interpolate_affine[v](distribution)
    # V ~ mixture_distribution
    #
    # thus,
    #
    # E[Var[Z | V]] = sum{ prob[d] Var[d] : d=0, ..., deg-1 }
    # Var[E[Z | V]] = sum{ prob[d] (Mean[d] - Mean)**2 : d=0, ..., deg-1 }
    v = array_ops.stack([
        # log(self.distribution.variance()) = log(Var[d]) = log(rate[d])
        self.distribution.log_rate,
        # log((Mean[d] - Mean)**2)
        2. * math_ops.log(
            math_ops.abs(self.distribution.mean()
                         - self._mean()[..., array_ops.newaxis])),
    ], axis=-1)
    return math_ops.reduce_logsumexp(
        self.mixture_distribution.logits[..., array_ops.newaxis] + v,
        axis=[-2, -1])


def concat_vectors(*args):
  """Concatenates input vectors, statically if possible."""
  args_ = [distribution_util.static_value(x) for x in args]
  if any(vec is None for vec in args_):
    return array_ops.concat(args, axis=0)
  return [val for vec in args_ for val in vec]
