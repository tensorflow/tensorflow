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
"""A Conditional Transformed Distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import conditional_distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import transformed_distribution
from tensorflow.python.ops.distributions import util as distribution_util


# pylint: disable=protected-access
_concat_vectors = transformed_distribution._concat_vectors
# pylint: enable=protected-access


__all__ = [
    "ConditionalTransformedDistribution",
]


_condition_kwargs_dict = {
    "bijector_kwargs": ("Python dictionary of arg names/values "
                        "forwarded to the bijector."),
    "distribution_kwargs": ("Python dictionary of arg names/values "
                            "forwarded to the distribution."),
}


class ConditionalTransformedDistribution(
    conditional_distribution.ConditionalDistribution,
    transformed_distribution.TransformedDistribution):
  """A TransformedDistribution that allows intrinsic conditioning."""

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _sample_n(self, n, seed=None,
                bijector_kwargs=None,
                distribution_kwargs=None):
    sample_shape = _concat_vectors(
        distribution_util.pick_vector(self._needs_rotation, self._empty, [n]),
        self._override_batch_shape,
        self._override_event_shape,
        distribution_util.pick_vector(self._needs_rotation, [n], self._empty))
    distribution_kwargs = distribution_kwargs or {}
    x = self.distribution.sample(sample_shape=sample_shape,
                                 seed=seed,
                                 **distribution_kwargs)
    x = self._maybe_rotate_dims(x)
    # We'll apply the bijector in the `_call_sample_n` function.
    return x

  def _call_sample_n(self, sample_shape, seed, name,
                     bijector_kwargs=None,
                     distribution_kwargs=None):
    # We override `_call_sample_n` rather than `_sample_n` so we can ensure that
    # the result of `self.bijector.forward` is not modified (and thus caching
    # works).
    with self._name_scope(name, values=[sample_shape]):
      sample_shape = ops.convert_to_tensor(
          sample_shape, dtype=dtypes.int32, name="sample_shape")
      sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, "sample_shape")

      # First, generate samples. We will possibly generate extra samples in the
      # event that we need to reinterpret the samples as part of the
      # event_shape.
      x = self._sample_n(n, seed, bijector_kwargs, distribution_kwargs)

      # Next, we reshape `x` into its final form. We do this prior to the call
      # to the bijector to ensure that the bijector caching works.
      batch_event_shape = array_ops.shape(x)[1:]
      final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
      x = array_ops.reshape(x, final_shape)

      # Finally, we apply the bijector's forward transformation. For caching to
      # work, it is imperative that this is the last modification to the
      # returned result.
      bijector_kwargs = bijector_kwargs or {}
      y = self.bijector.forward(x, **bijector_kwargs)
      y = self._set_sample_static_shape(y, sample_shape)

      return y

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _log_prob(self, y, bijector_kwargs=None, distribution_kwargs=None):
    # For caching to work, it is imperative that the bijector is the first to
    # modify the input.
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    ildj = self.bijector.inverse_log_det_jacobian(y, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_log_prob_for_one_fiber(y, x, ildj,
                                                 distribution_kwargs)

    lp_on_fibers = [
        self._finish_log_prob_for_one_fiber(y, x_i, ildj_i, distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return math_ops.reduce_logsumexp(array_ops.stack(lp_on_fibers), axis=0)

  def _finish_log_prob_for_one_fiber(self, y, x, ildj, distribution_kwargs):
    """Finish computation of log_prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    log_prob = self.distribution.log_prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      log_prob = math_ops.reduce_sum(log_prob, self._reduce_event_indices)
    return ildj + log_prob

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _prob(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    ildj = self.bijector.inverse_log_det_jacobian(y, **bijector_kwargs)
    if self.bijector._is_injective:  # pylint: disable=protected-access
      return self._finish_prob_for_one_fiber(y, x, ildj, distribution_kwargs)

    prob_on_fibers = [
        self._finish_prob_for_one_fiber(y, x_i, ildj_i, distribution_kwargs)
        for x_i, ildj_i in zip(x, ildj)]
    return sum(prob_on_fibers)

  def _finish_prob_for_one_fiber(self, y, x, ildj, distribution_kwargs):
    """Finish computation of prob on one element of the inverse image."""
    x = self._maybe_rotate_dims(x, rotate_right=True)
    prob = self.distribution.prob(x, **distribution_kwargs)
    if self._is_maybe_event_override:
      prob = math_ops.reduce_prod(prob, self._reduce_event_indices)
    return math_ops.exp(ildj) * prob

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _log_cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_cdf is not implemented when "
                                "bijector is not injective.")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("cdf is not implemented when "
                                "bijector is not injective.")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _log_survival_function(self, y,
                             bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("log_survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("log_survival_function is not implemented when "
                                "bijector is not injective.")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_survival_function(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _survival_function(self, y,
                         bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("survival_function is not implemented when "
                                "overriding event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("survival_function is not implemented when "
                                "bijector is not injective.")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.survival_function(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _quantile(self, value, bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("quantile is not implemented when overriding "
                                "event_shape")
    if not self.bijector._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError("quantile is not implemented when "
                                "bijector is not injective.")
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    # x_q is the "qth quantile" of X iff q = P[X <= x_q].  Now, since X =
    # g^{-1}(Y), q = P[X <= x_q] = P[g^{-1}(Y) <= x_q] = P[Y <= g(x_q)],
    # implies the qth quantile of Y is g(x_q).
    inv_cdf = self.distribution.quantile(value, **distribution_kwargs)
    return self.bijector.forward(inv_cdf, **bijector_kwargs)
