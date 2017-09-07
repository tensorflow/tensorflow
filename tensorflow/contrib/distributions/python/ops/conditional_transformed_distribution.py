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
                bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    sample_shape = _concat_vectors(
        distribution_util.pick_vector(self._needs_rotation, self._empty, [n]),
        self._override_batch_shape,
        self._override_event_shape,
        distribution_util.pick_vector(self._needs_rotation, [n], self._empty))
    x = self.distribution.sample(sample_shape=sample_shape, seed=seed,
                                 **distribution_kwargs)
    x = self._maybe_rotate_dims(x)
    return self.bijector.forward(x, **bijector_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _log_prob(self, y, bijector_kwargs=None, distribution_kwargs=None):
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    ildj = self.bijector.inverse_log_det_jacobian(y, **bijector_kwargs)
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
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.log_cdf(x, **distribution_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict=_condition_kwargs_dict)
  def _cdf(self, y, bijector_kwargs=None, distribution_kwargs=None):
    if self._is_maybe_event_override:
      raise NotImplementedError("cdf is not implemented when overriding "
                                "event_shape")
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
    bijector_kwargs = bijector_kwargs or {}
    distribution_kwargs = distribution_kwargs or {}
    x = self.bijector.inverse(y, **bijector_kwargs)
    return self.distribution.survival_function(x, **distribution_kwargs)
