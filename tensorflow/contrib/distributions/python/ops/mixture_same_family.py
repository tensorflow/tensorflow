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
"""The same-family Mixture distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util


class MixtureSameFamily(distribution.Distribution):
  """Mixture (same-family) distribution.

  The `MixtureSameFamily` distribution implements a (batch of) mixture
  distribution where all components are from different parameterizations of the
  same distribution type. It is parameterized by a `Categorical` "selecting
  distribution" (over `k` components) and a components distribution, i.e., a
  `Distribution` with a rightmost batch shape (equal to `[k]`) which indexes
  each (batch of) component.

  #### Examples

  ```python
  import matplotlib.pyplot as plt
  ds = tf.contrib.distributions

  ### Create a mixture of two scalar Gaussians:

  gm = ds.MixtureSameFamily(
      mixture_distribution=ds.Categorical(
          probs=[0.3, 0.7]),
      components_distribution=ds.Normal(
        loc=[-1., 1],       # One for each component.
        scale=[0.1, 0.5]))  # And same here.

  gm.mean()
  # ==> 0.4

  gm.variance()
  # ==> 1.018

  # Plot PDF.
  x = np.linspace(-2., 3., int(1e4), dtype=np.float32)
  plt.plot(x, gm.prob(x).eval());

  ### Create a mixture of two Bivariate Gaussians:

  gm = ds.MixtureSameFamily(
      mixture_distribution=ds.Categorical(
          probs=[0.3, 0.7]),
      components_distribution=ds.MultivariateNormalDiag(
          loc=[[-1., 1],  # component 1
               [1, -1]],  # component 2
          scale_identity_multiplier=[.3, .6]))

  gm.mean()
  # ==> array([ 0.4, -0.4], dtype=float32)

  gm.covariance()
  # ==> array([[ 1.119, -0.84],
  #            [-0.84,  1.119]], dtype=float32)

  # Plot PDF contours.
  def meshgrid(x, y=x):
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
  grid = meshgrid(np.linspace(-2, 2, 100, dtype=np.float32))
  plt.contour(grid[..., 0], grid[..., 1], gm.prob(grid).eval());

  ```

  """

  def __init__(self,
               mixture_distribution,
               components_distribution,
               validate_args=False,
               allow_nan_stats=True,
               name="MixtureSameFamily"):
    """Construct a `MixtureSameFamily` distribution.

    Args:
      mixture_distribution: `tf.distributions.Categorical`-like instance.
        Manages the probability of selecting components. The number of
        categories must match the rightmost batch dimension of the
        `components_distribution`. Must have either scalar `batch_shape` or
        `batch_shape` matching `components_distribution.batch_shape[:-1]`.
      components_distribution: `tf.distributions.Distribution`-like instance.
        Right-most batch dimension indexes components.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: `if not mixture_distribution.dtype.is_integer`.
      ValueError: if mixture_distribution does not have scalar `event_shape`.
      ValueError: if `mixture_distribution.batch_shape` and
        `components_distribution.batch_shape[:-1]` are both fully defined and
        the former is neither scalar nor equal to the latter.
      ValueError: if `mixture_distribution` categories does not equal
        `components_distribution` rightmost batch shape.
    """
    parameters = locals()
    with ops.name_scope(name):
      self._mixture_distribution = mixture_distribution
      self._components_distribution = components_distribution
      self._runtime_assertions = []

      s = components_distribution.event_shape_tensor()
      self._event_ndims = (s.shape[0].value
                           if s.shape.with_rank_at_least(1)[0].value is not None
                           else array_ops.shape(s)[0])

      if not mixture_distribution.dtype.is_integer:
        raise ValueError(
            "`mixture_distribution.dtype` ({}) is not over integers".format(
                mixture_distribution.dtype.name))

      if (mixture_distribution.event_shape.ndims is not None
          and mixture_distribution.event_shape.ndims != 0):
        raise ValueError("`mixture_distribution` must have scalar `event_dim`s")
      elif validate_args:
        self._runtime_assertions += [
            control_flow_ops.assert_has_rank(
                mixture_distribution.event_shape_tensor(), 0,
                message="`mixture_distribution` must have scalar `event_dim`s"),
        ]

      mdbs = mixture_distribution.batch_shape
      cdbs = components_distribution.batch_shape.with_rank_at_least(1)[:-1]
      if mdbs.is_fully_defined() and cdbs.is_fully_defined():
        if mdbs.ndims != 0 and mdbs != cdbs:
          raise ValueError(
              "`mixture_distribution.batch_shape` (`{}`) is not "
              "compatible with `components_distribution.batch_shape` "
              "(`{}`)".format(mdbs.as_list(), cdbs.as_list()))
      elif validate_args:
        mdbs = mixture_distribution.batch_shape_tensor()
        cdbs = components_distribution.batch_shape_tensor()[:-1]
        self._runtime_assertions += [
            control_flow_ops.assert_equal(
                distribution_util.pick_vector(
                    mixture_distribution.is_scalar_batch(), cdbs, mdbs),
                cdbs,
                message=(
                    "`mixture_distribution.batch_shape` is not "
                    "compatible with `components_distribution.batch_shape`"))]

      km = mixture_distribution.logits.shape.with_rank_at_least(1)[-1].value
      kc = components_distribution.batch_shape.with_rank_at_least(1)[-1].value
      if km is not None and kc is not None and km != kc:
        raise ValueError("`mixture_distribution components` ({}) does not "
                         "equal `components_distribution.batch_shape[-1]` "
                         "({})".format(km, kc))
      elif validate_args:
        km = array_ops.shape(mixture_distribution.logits)[-1]
        kc = components_distribution.batch_shape_tensor()[-1]
        self._runtime_assertions += [
            control_flow_ops.assert_equal(
                km, kc,
                message=("`mixture_distribution components` does not equal "
                         "`components_distribution.batch_shape[-1:]`")),
        ]
      elif km is None:
        km = array_ops.shape(mixture_distribution.logits)[-1]

      self._num_components = km

      super(MixtureSameFamily, self).__init__(
          dtype=self._components_distribution.dtype,
          reparameterization_type=distribution.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=(
              self._mixture_distribution._graph_parents  # pylint: disable=protected-access
              + self._components_distribution._graph_parents),  # pylint: disable=protected-access
          name=name)

  @property
  def mixture_distribution(self):
    return self._mixture_distribution

  @property
  def components_distribution(self):
    return self._components_distribution

  def _batch_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.components_distribution.batch_shape_tensor()[:-1]

  def _batch_shape(self):
    return self.components_distribution.batch_shape.with_rank_at_least(1)[:-1]

  def _event_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.components_distribution.event_shape_tensor()

  def _event_shape(self):
    return self.components_distribution.event_shape

  def _sample_n(self, n, seed):
    with ops.control_dependencies(self._runtime_assertions):
      x = self.components_distribution.sample(n)             # [n, B, k, E]
      # TODO(jvdillon): Consider using tf.gather (by way of index unrolling).
      npdt = x.dtype.as_numpy_dtype
      mask = array_ops.one_hot(
          indices=self.mixture_distribution.sample(n),       # [n, B]
          depth=self._num_components,                        # == k
          on_value=np.ones([], dtype=npdt),
          off_value=np.zeros([], dtype=npdt))                # [n, B, k]
      mask = self._pad_mix_dims(mask)                        # [n, B, k, [1]*e]
      return math_ops.reduce_sum(
          x * mask, axis=-1 - self._event_ndims)             # [n, B, E]

  def _log_prob(self, x):
    with ops.control_dependencies(self._runtime_assertions):
      x = self._pad_sample_dims(x)
      log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
      log_mix_prob = nn_ops.log_softmax(
          self.mixture_distribution.logits, dim=-1)          # [B, k]
      return math_ops.reduce_logsumexp(
          log_prob_x + log_mix_prob, axis=-1)                # [S, B]

  def _mean(self):
    with ops.control_dependencies(self._runtime_assertions):
      probs = self._pad_mix_dims(
          self.mixture_distribution.probs)                   # [B, k, [1]*e]
      return math_ops.reduce_sum(
          probs * self.components_distribution.mean(),
          axis=-1 - self._event_ndims)                       # [B, E]

  def _variance(self):
    with ops.control_dependencies(self._runtime_assertions):
      # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
      probs = self._pad_mix_dims(
          self.mixture_distribution.probs)                   # [B, k, [1]*e]
      mean_cond_var = math_ops.reduce_sum(
          probs * self.components_distribution.variance(),
          axis=-1 - self._event_ndims)                       # [B, E]
      var_cond_mean = math_ops.reduce_sum(
          probs * math_ops.squared_difference(
              self.components_distribution.mean(),
              self._pad_sample_dims(self._mean())),
          axis=-1 - self._event_ndims)                       # [B, E]
      return mean_cond_var + var_cond_mean                   # [B, E]

  def _covariance(self):
    static_event_ndims = self.event_shape.ndims
    if static_event_ndims != 1:
      # Covariance is defined only for vector distributions.
      raise NotImplementedError("covariance is not implemented")

    with ops.control_dependencies(self._runtime_assertions):
      # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
      probs = self._pad_mix_dims(self._pad_mix_dims(
          self.mixture_distribution.probs))                  # [B, k, 1, 1]
      mean_cond_var = math_ops.reduce_sum(
          probs * self.components_distribution.covariance(),
          axis=-3)                                           # [B, e, e]
      var_cond_mean = math_ops.reduce_sum(
          probs * _outer_squared_difference(
              self.components_distribution.mean(),
              self._pad_sample_dims(self._mean())),
          axis=-3)                                           # [B, e, e]
      return mean_cond_var + var_cond_mean                   # [B, e, e]

  def _pad_sample_dims(self, x):
    with ops.name_scope("pad_sample_dims", values=[x]):
      ndims = x.shape.ndims if x.shape.ndims is not None else array_ops.rank(x)
      shape = array_ops.shape(x)
      d = ndims - self._event_ndims
      x = array_ops.reshape(x, shape=array_ops.concat([
          shape[:d], [1], shape[d:]], axis=0))
      return x

  def _pad_mix_dims(self, x):
    with ops.name_scope("pad_mix_dims", values=[x]):
      def _get_ndims(d):
        if d.batch_shape.ndims is not None:
          return d.batch_shape.ndims
        return array_ops.shape(d.batch_shape_tensor())[0]
      dist_batch_ndims = _get_ndims(self)
      cat_batch_ndims = _get_ndims(self.mixture_distribution)
      bnd = distribution_util.pick_vector(
          self.mixture_distribution.is_scalar_batch(),
          [dist_batch_ndims], [cat_batch_ndims])[0]
      s = array_ops.shape(x)
      x = array_ops.reshape(x, shape=array_ops.concat([
          s[:-1],
          array_ops.ones([bnd], dtype=dtypes.int32),
          s[-1:],
          array_ops.ones([self._event_ndims], dtype=dtypes.int32),
      ], axis=0))
      return x


def _outer_squared_difference(x, y):
  """Convenience function analogous to tf.squared_difference."""
  z = x - y
  return z[..., array_ops.newaxis, :] * z[..., array_ops.newaxis]
