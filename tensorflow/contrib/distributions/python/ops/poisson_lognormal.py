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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import categorical as categorical_lib
from tensorflow.python.ops.distributions import distribution as distribution_lib


__all__ = [
    "PoissonLogNormalQuadratureCompound",
]


class PoissonLogNormalQuadratureCompound(distribution_lib.Distribution):
  """`PoissonLogNormalQuadratureCompound` distribution.

  The `PoissonLogNormalQuadratureCompound` is an approximation to a
  Poisson-LogNormal [compound distribution](
  https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e.,

  ```none
  p(k|loc, scale)
  = int_{R_+} dl LogNormal(l | loc, scale) Poisson(k | l)
  = int_{R} dz ((lambda(z) sqrt(2) scale)
                * exp(-z**2) / (lambda(z) sqrt(2 pi) sigma)
                * Poisson(k | lambda(z)))
  = int_{R} dz exp(-z**2) / sqrt(pi) Poisson(k | lambda(z))
  approx= sum{ prob[d] Poisson(k | lambda(grid[d])) : d=0, ..., deg-1 }
  ```

  where `lambda(z) = exp(sqrt(2) scale z + loc)` and the `prob,grid` terms
  are from [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  [Gauss--Hermite quadrature](
  https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)). Note that
  the second line made the substitution:
  `z(l) = (log(l) - loc) / (sqrt(2) scale)` which implies `lambda(z)` [above]
  and `dl = sqrt(2) scale lambda(z) dz`

  In the non-approximation case, a draw from the LogNormal prior represents the
  Poisson rate parameter. Unfortunately, the non-approximate distribution lacks
  an analytical probability density function (pdf). Therefore the
  `PoissonLogNormalQuadratureCompound` class implements an approximation based
  on [numerical quadrature](
  https://en.wikipedia.org/wiki/Numerical_integration) (default:
  [Gauss--Hermite quadrature](
  https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)).

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
  [Gauss--Hermite quadrature](
  https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)) we can
  redefine the distribution to be a parameter-less convex combination of `deg`
  different Poisson samples.

  That is, defined over positive integers, this distribution is parameterized
  by a (batch of) `loc` and `scale` scalars.

  The probability density function (pdf) is,

  ```none
  pdf(k | loc, scale, deg)
    = sum{ prob[d] Poisson(k | lambda=exp(sqrt(2) scale grid[d] + loc))
          : d=0, ..., deg-1 }
  ```

  where, [e.g., `grid, w = numpy.polynomial.hermite.hermgauss(deg)`](
  https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polynomial.hermite.hermgauss.html)
  and `prob = w / sqrt(pi)`.

  #### Examples

  ```python
  ds = tf.contrib.distributions
  # Create two batches of PoissonLogNormalQuadratureCompounds, one with
  # prior `loc = 0.` and another with `loc = 1.` In both cases `scale = 1.`
  pln = ds.PoissonLogNormalQuadratureCompound(
      loc=[0., -0.5],
      scale=1.,
      quadrature_grid_and_probs=(
        np.polynomial.hermite.hermgauss(deg=10)),
      validate_args=True)
  """

  def __init__(self,
               loc,
               scale,
               quadrature_grid_and_probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="PoissonLogNormalQuadratureCompound"):
    """Constructs the PoissonLogNormalQuadratureCompound on `R**k`.

    Args:
      loc: `float`-like (batch of) scalar `Tensor`; the location parameter of
        the LogNormal prior.
      scale: `float`-like (batch of) scalar `Tensor`; the scale parameter of
        the LogNormal prior.
      quadrature_grid_and_probs: Python pair of `float`-like `Tensor`s
        representing the sample points and the corresponding (possibly
        normalized) weight.  When `None`, defaults to:
        `np.polynomial.hermite.hermgauss(deg=8)`.
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
      TypeError: if `loc.dtype != scale[0].dtype`.
    """
    parameters = locals()
    with ops.name_scope(name, values=[loc, scale]):
      loc = ops.convert_to_tensor(loc, name="loc")
      self._loc = loc

      scale = ops.convert_to_tensor(scale, name="scale")
      self._scale = scale

      dtype = loc.dtype.base_dtype
      if dtype != scale.dtype.base_dtype:
        raise TypeError(
            "loc.dtype(\"{}\") does not match scale.dtype(\"{}\")".format(
                loc.dtype.name, scale.dtype.name))

      grid, probs = distribution_util.process_quadrature_grid_and_probs(
          quadrature_grid_and_probs, dtype, validate_args)
      self._quadrature_grid = grid
      self._quadrature_probs = probs
      self._quadrature_size = distribution_util.dimension_size(probs, axis=0)

      self._mixture_distribution = categorical_lib.Categorical(
          logits=math_ops.log(self._quadrature_probs),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      # The following maps the broadcast of `loc` and `scale` to each grid
      # point, i.e., we are creating several log-rates that correspond to the
      # different Gauss-Hermite quadrature points and (possible) batches of
      # `loc` and `scale`.
      self._log_rate = (loc[..., array_ops.newaxis]
                        + np.sqrt(2.) * scale[..., array_ops.newaxis] * grid)

      self._distribution = poisson_lib.Poisson(
          log_rate=self._log_rate,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      super(PoissonLogNormalQuadratureCompound, self).__init__(
          dtype=dtype,
          reparameterization_type=distribution_lib.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[loc, scale],
          name=name)

  @property
  def mixture_distribution(self):
    """Distribution which randomly selects a Poisson with Gauss-Hermite rate."""
    return self._mixture_distribution

  @property
  def distribution(self):
    """Base Poisson parameterized by a Gauss-Hermite grid of rates."""
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
  def quadrature_grid(self):
    """Quadrature grid points."""
    return self._quadrature_grid

  @property
  def quadrature_probs(self):
    """Quadrature normalized weights."""
    return self._quadrature_probs

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc),
        array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.shape,
        self.scale.shape)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # Get ids as a [n, batch_size]-shaped matrix, unless batch_shape=[] then get
    # ids as a [n]-shaped vector.
    batch_size = (np.prod(self.batch_shape.as_list(), dtype=np.int32)
                  if self.batch_shape.is_fully_defined()
                  else math_ops.reduce_prod(self.batch_shape_tensor()))
    ids = self._mixture_distribution.sample(
        sample_shape=concat_vectors(
            [n],
            distribution_util.pick_vector(
                self.is_scalar_batch(),
                np.int32([]),
                [batch_size])),
        seed=distribution_util.gen_new_seed(
            seed, "poisson_lognormal_quadrature_compound"))
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
            self.mixture_distribution.logits + self._log_rate,
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
    # V ~ mixture_distrubution
    #
    # thus,
    #
    # E[Var[Z | V]] = sum{ prob[d] Var[d] : d=0, ..., deg-1 }
    # Var[E[Z | V]] = sum{ prob[d] (Mean[d] - Mean)**2 : d=0, ..., deg-1 }
    v = array_ops.stack([
        # log(self.distribution.variance()) = log(Var[d]) = log(rate[d])
        self._log_rate,
        # log((Mean[d] - Mean)**2)
        2. * math_ops.log(
            math_ops.abs(self.distribution.mean()
                         - self._mean()[..., array_ops.newaxis])),
    ], axis=-1)
    return math_ops.reduce_logsumexp(
        self.mixture_distribution.logits[..., array_ops.newaxis] + v,
        axis=[-2, -1])


def static_value(x):
  """Returns the static value of a `Tensor` or `None`."""
  return tensor_util.constant_value(ops.convert_to_tensor(x))


def concat_vectors(*args):
  """Concatenates input vectors, statically if possible."""
  args_ = [static_value(x) for x in args]
  if any(vec is None for vec in args_):
    return array_ops.concat(args, axis=0)
  return [val for vec in args_ for val in vec]
