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
"""The Bernoulli distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util.tf_export import tf_export


@tf_export("distributions.Bernoulli")
class Bernoulli(distribution.Distribution):
  """Bernoulli distribution.

  The Bernoulli distribution with `probs` parameter, i.e., the probability of a
  `1` outcome (vs a `0` outcome).
  """

  def __init__(self,
               logits=None,
               probs=None,
               dtype=dtypes.int32,
               validate_args=False,
               allow_nan_stats=True,
               name="Bernoulli"):
    """Construct Bernoulli distributions.

    Args:
      logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
        entry in the `Tensor` parametrizes an independent Bernoulli distribution
        where the probability of an event is sigmoid(logits). Only one of
        `logits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a `1`
        event. Each entry in the `Tensor` parameterizes an independent
        Bernoulli distribution. Only one of `logits` or `probs` should be passed
        in.
      dtype: The type of the event samples. Default: `int32`.
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
      ValueError: If p and logits are passed, or if neither are passed.
    """
    parameters = locals()
    with ops.name_scope(name):
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          name=name)
    super(Bernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits, self._probs],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"logits": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    return self._logits

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    return self._probs

  def _batch_shape_tensor(self):
    return array_ops.shape(self._logits)

  def _batch_shape(self):
    return self._logits.get_shape()

  def _event_shape_tensor(self):
    return array_ops.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    new_shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    uniform = random_ops.random_uniform(
        new_shape, seed=seed, dtype=self.probs.dtype)
    sample = math_ops.less(uniform, self.probs)
    return math_ops.cast(sample, self.dtype)

  def _log_prob(self, event):
    if self.validate_args:
      event = distribution_util.embed_check_integer_casting_closed(
          event, target_dtype=dtypes.bool)

    # TODO(jaana): The current sigmoid_cross_entropy_with_logits has
    # inconsistent behavior for logits = inf/-inf.
    event = math_ops.cast(event, self.logits.dtype)
    logits = self.logits
    # sigmoid_cross_entropy_with_logits doesn't broadcast shape,
    # so we do this here.

    def _broadcast(logits, event):
      return (array_ops.ones_like(event) * logits,
              array_ops.ones_like(logits) * event)

    if not (event.get_shape().is_fully_defined() and
            logits.get_shape().is_fully_defined() and
            event.get_shape() == logits.get_shape()):
      logits, event = _broadcast(logits, event)
    return -nn.sigmoid_cross_entropy_with_logits(labels=event, logits=logits)

  def _entropy(self):
    return (-self.logits * (math_ops.sigmoid(self.logits) - 1) +
            nn.softplus(-self.logits))

  def _mean(self):
    return array_ops.identity(self.probs)

  def _variance(self):
    return self._mean() * (1. - self.probs)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return math_ops.cast(self.probs > 0.5, self.dtype)


@kullback_leibler.RegisterKL(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Bernoulli.

  Args:
    a: instance of a Bernoulli distribution object.
    b: instance of a Bernoulli distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_bernoulli_bernoulli".

  Returns:
    Batchwise KL(a || b)
  """
  with ops.name_scope(name, "kl_bernoulli_bernoulli",
                      values=[a.logits, b.logits]):
    delta_probs0 = nn.softplus(-b.logits) - nn.softplus(-a.logits)
    delta_probs1 = nn.softplus(b.logits) - nn.softplus(a.logits)
    return (math_ops.sigmoid(a.logits) * delta_probs0
            + math_ops.sigmoid(-a.logits) * delta_probs1)
