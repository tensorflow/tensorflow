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
"""The Independent distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.ops.distributions import kullback_leibler


class Independent(distribution_lib.Distribution):
  """Independent distribution from batch of distributions.

  This distribution is useful for regarding a collection of independent,
  non-identical distributions as a single random variable. For example, the
  `Independent` distribution composed of a collection of `Bernoulli`
  distributions might define a distribution over an image (where each
  `Bernoulli` is a distribution over each pixel).

  More precisely, a collection of `B` (independent) `E`-variate random variables
  (rv) `{X_1, ..., X_B}`, can be regarded as a `[B, E]`-variate random variable
  `(X_1, ..., X_B)` with probability
  `p(x_1, ..., x_B) = p_1(x_1) * ... * p_B(x_B)` where `p_b(X_b)` is the
  probability of the `b`-th rv. More generally `B, E` can be arbitrary shapes.

  Similarly, the `Independent` distribution specifies a distribution over `[B,
  E]`-shaped events. It operates by reinterpreting the rightmost batch dims as
  part of the event dimensions. The `reinterpreted_batch_ndims` parameter
  controls the number of batch dims which are absorbed as event dims;
  `reinterpreted_batch_ndims < len(batch_shape)`.  For example, the `log_prob`
  function entails a `reduce_sum` over the rightmost `reinterpreted_batch_ndims`
  after calling the base distribution's `log_prob`.  In other words, since the
  batch dimension(s) index independent distributions, the resultant multivariate
  will have independent components.

  #### Mathematical Details

  The probability function is,

  ```none
  prob(x; reinterpreted_batch_ndims) = tf.reduce_prod(
      dist.prob(x),
      axis=-1-range(reinterpreted_batch_ndims))
  ```

  #### Examples

  ```python
  tfd = tf.contrib.distributions

  # Make independent distribution from a 2-batch Normal.
  ind = tfd.Independent(
      distribution=tfd.Normal(loc=[-1., 1], scale=[0.1, 0.5]),
      reinterpreted_batch_ndims=1)

  # All batch dims have been "absorbed" into event dims.
  ind.batch_shape  # ==> []
  ind.event_shape  # ==> [2]

  # Make independent distribution from a 2-batch bivariate Normal.
  ind = tfd.Independent(
      distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., 1], [1, -1]],
          scale_identity_multiplier=[1., 0.5]),
      reinterpreted_batch_ndims=1)

  # All batch dims have been "absorbed" into event dims.
  ind.batch_shape  # ==> []
  ind.event_shape  # ==> [2, 2]
  ```

  """

  def __init__(
      self, distribution, reinterpreted_batch_ndims=None,
      validate_args=False, name=None):
    """Construct a `Independent` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims
        which will be regarded as event dims. When `None` all but the first
        batch axis (batch axis 0) will be transferred to event dimensions
        (analogous to `tf.layers.flatten`).
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `Independent + distribution.name`.

    Raises:
      ValueError: if `reinterpreted_batch_ndims` exceeds
        `distribution.batch_ndims`
    """
    parameters = locals()
    name = name or "Independent" + distribution.name
    self._distribution = distribution
    with ops.name_scope(name) as name:
      if reinterpreted_batch_ndims is None:
        reinterpreted_batch_ndims = self._get_default_reinterpreted_batch_ndims(
            distribution)
      reinterpreted_batch_ndims = ops.convert_to_tensor(
          reinterpreted_batch_ndims,
          dtype=dtypes.int32,
          name="reinterpreted_batch_ndims")
      self._reinterpreted_batch_ndims = reinterpreted_batch_ndims
      self._static_reinterpreted_batch_ndims = tensor_util.constant_value(
          reinterpreted_batch_ndims)
      if self._static_reinterpreted_batch_ndims is not None:
        self._reinterpreted_batch_ndims = self._static_reinterpreted_batch_ndims
      super(Independent, self).__init__(
          dtype=self._distribution.dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          graph_parents=(
              [reinterpreted_batch_ndims] +
              distribution._graph_parents),  # pylint: disable=protected-access
          name=name)
      self._runtime_assertions = self._make_runtime_assertions(
          distribution, reinterpreted_batch_ndims, validate_args)

  @property
  def distribution(self):
    return self._distribution

  @property
  def reinterpreted_batch_ndims(self):
    return self._reinterpreted_batch_ndims

  def _batch_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      batch_shape = self.distribution.batch_shape_tensor()
      batch_ndims = (batch_shape.shape[0].value
                     if batch_shape.shape.with_rank_at_least(1)[0].value
                     else array_ops.shape(batch_shape)[0])
      return batch_shape[:batch_ndims - self.reinterpreted_batch_ndims]

  def _batch_shape(self):
    batch_shape = self.distribution.batch_shape
    if (self._static_reinterpreted_batch_ndims is None
        or batch_shape.ndims is None):
      return tensor_shape.TensorShape(None)
    d = batch_shape.ndims - self._static_reinterpreted_batch_ndims
    return batch_shape[:d]

  def _event_shape_tensor(self):
    with ops.control_dependencies(self._runtime_assertions):
      batch_shape = self.distribution.batch_shape_tensor()
      batch_ndims = (batch_shape.shape[0].value
                     if batch_shape.shape.with_rank_at_least(1)[0].value
                     else array_ops.shape(batch_shape)[0])
      return array_ops.concat([
          batch_shape[batch_ndims - self.reinterpreted_batch_ndims:],
          self.distribution.event_shape_tensor(),
      ], axis=0)

  def _event_shape(self):
    batch_shape = self.distribution.batch_shape
    if (self._static_reinterpreted_batch_ndims is None
        or batch_shape.ndims is None):
      return tensor_shape.TensorShape(None)
    d = batch_shape.ndims - self._static_reinterpreted_batch_ndims
    return batch_shape[d:].concatenate(self.distribution.event_shape)

  def _sample_n(self, n, seed):
    with ops.control_dependencies(self._runtime_assertions):
      return self.distribution.sample(sample_shape=n, seed=seed)

  def _log_prob(self, x):
    with ops.control_dependencies(self._runtime_assertions):
      return self._reduce_sum(self.distribution.log_prob(x))

  def _entropy(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self._reduce_sum(self.distribution.entropy())

  def _mean(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.distribution.mean()

  def _variance(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.distribution.variance()

  def _stddev(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.distribution.stddev()

  def _mode(self):
    with ops.control_dependencies(self._runtime_assertions):
      return self.distribution.mode()

  def _make_runtime_assertions(
      self, distribution, reinterpreted_batch_ndims, validate_args):
    assertions = []
    static_reinterpreted_batch_ndims = tensor_util.constant_value(
        reinterpreted_batch_ndims)
    batch_ndims = distribution.batch_shape.ndims
    if batch_ndims is not None and static_reinterpreted_batch_ndims is not None:
      if static_reinterpreted_batch_ndims > batch_ndims:
        raise ValueError("reinterpreted_batch_ndims({}) cannot exceed "
                         "distribution.batch_ndims({})".format(
                             static_reinterpreted_batch_ndims, batch_ndims))
    elif validate_args:
      batch_shape = distribution.batch_shape_tensor()
      batch_ndims = (
          batch_shape.shape[0].value
          if batch_shape.shape.with_rank_at_least(1)[0].value is not None
          else array_ops.shape(batch_shape)[0])
      assertions.append(check_ops.assert_less_equal(
          reinterpreted_batch_ndims, batch_ndims,
          message=("reinterpreted_batch_ndims cannot exceed "
                   "distribution.batch_ndims")))
    return assertions

  def _reduce_sum(self, stat):
    if self._static_reinterpreted_batch_ndims is None:
      range_ = math_ops.range(self._reinterpreted_batch_ndims)
    else:
      range_ = np.arange(self._static_reinterpreted_batch_ndims)
    return math_ops.reduce_sum(stat, axis=-1-range_)

  def _get_default_reinterpreted_batch_ndims(self, distribution):
    """Computes the default value for reinterpreted_batch_ndim __init__ arg."""
    ndims = distribution.batch_shape.ndims
    if ndims is None:
      which_maximum = math_ops.maximum
      ndims = array_ops.shape(distribution.batch_shape_tensor())[0]
    else:
      which_maximum = np.maximum
    return which_maximum(0, ndims - 1)


@kullback_leibler.RegisterKL(Independent, Independent)
def _kl_independent(a, b, name="kl_independent"):
  """Batched KL divergence `KL(a || b)` for Independent distributions.

  We can leverage the fact that
  ```
  KL(Independent(a) || Independent(b)) = sum(KL(a || b))
  ```
  where the sum is over the `reinterpreted_batch_ndims`.

  Args:
    a: Instance of `Independent`.
    b: Instance of `Independent`.
    name: (optional) name to use for created ops. Default "kl_independent".

  Returns:
    Batchwise `KL(a || b)`.

  Raises:
    ValueError: If the event space for `a` and `b`, or their underlying
      distributions don't match.
  """
  p = a.distribution
  q = b.distribution

  # The KL between any two (non)-batched distributions is a scalar.
  # Given that the KL between two factored distributions is the sum, i.e.
  # KL(p1(x)p2(y) || q1(x)q2(y)) = KL(p1 || q1) + KL(q1 || q2), we compute
  # KL(p || q) and do a `reduce_sum` on the reinterpreted batch dimensions.
  if a.event_shape.is_fully_defined() and b.event_shape.is_fully_defined():
    if a.event_shape == b.event_shape:
      if p.event_shape == q.event_shape:
        num_reduce_dims = a.event_shape.ndims - p.event_shape.ndims
        reduce_dims = [-i - 1 for i in range(0, num_reduce_dims)]

        return math_ops.reduce_sum(
            kullback_leibler.kl_divergence(p, q, name=name), axis=reduce_dims)
      else:
        raise NotImplementedError("KL between Independents with different "
                                  "event shapes not supported.")
    else:
      raise ValueError("Event shapes do not match.")
  else:
    with ops.control_dependencies([
        check_ops.assert_equal(a.event_shape_tensor(), b.event_shape_tensor()),
        check_ops.assert_equal(p.event_shape_tensor(), q.event_shape_tensor())
    ]):
      num_reduce_dims = (
          array_ops.shape(a.event_shape_tensor()[0]) -
          array_ops.shape(p.event_shape_tensor()[0]))
      reduce_dims = math_ops.range(-num_reduce_dims - 1, -1, 1)
      return math_ops.reduce_sum(
          kullback_leibler.kl_divergence(p, q, name=name), axis=reduce_dims)
