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
"""Stochastic gradient estimators.

These functions are meant to be used in conjuction with `StochasticTensor`
(`loss_fn` parameter) and `surrogate_loss`.

See Gradient Estimation Using Stochastic Computation Graphs
(http://arxiv.org/abs/1506.05254) by Schulman et al., eq. 1 and section 4, for
mathematical details.

## Score function estimator

The score function is an unbiased estimator of the gradient of `E_p(x)[f(x)]`,
where `f(x)` can be considered to be a "loss" term. It is computed as
`E_p(x)[f(x) grad(log p(x))]`. A constant `b`, referred to here as the
"baseline", can be subtracted from `f(x)` without affecting the expectation. The
term `(f(x) - b)` is referred to here as the "advantage".

Note that the methods defined in this module actually compute the integrand of
the score function, such that when taking the gradient, the true score function
is computed.

@@score_function
@@get_score_function_with_baseline
@@get_score_function_with_constant_baseline
@@get_score_function_with_advantage

## Baseline functions

Baselines reduce the variance of Monte Carlo estimate of an expectation. The
baseline for a stochastic node can be a function of all non-influenced nodes
(see section 4 of Schulman et al., linked above). Baselines are also known as
"control variates."

In the context of a MC estimate of `E_p(x)[f(x) - b]`, baseline functions have
the signature `(st, fx) => Tensor`, where `st` is a `StochasticTensor` backed by
the distribution `p(x)` and `fx` is the influenced loss.

@@get_mean_baseline

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import training
from tensorflow.python.util.all_util import make_all


def score_function(dist_tensor, value, loss, baseline=None,
                   name="ScoreFunction"):
  """Score function estimator.

  Computes the integrand of the score function with a baseline:
  `p.log_prob(value) * (loss - baseline)`.

  It will add a `stop_gradient` to the advantage `(loss - baseline)`.

  Args:
    dist_tensor: `DistributionTensor` p(x).
    value: `Tensor` x. Samples from p(x).
    loss: `Tensor`.
    baseline: `Tensor` broadcastable to `loss`.
    name: name to prepend ops with.

  Returns:
    `Tensor` `p.log_prob(x) * (loss - b)`. Taking the gradient yields the score
    function estimator.
  """
  with ops.name_scope(name, values=[value, loss, baseline]):
    value = ops.convert_to_tensor(value)
    loss = ops.convert_to_tensor(loss)
    if baseline is not None:
      baseline = ops.convert_to_tensor(baseline)
      advantage = loss - baseline
    else:
      advantage = loss

    advantage = array_ops.stop_gradient(advantage)
    return dist_tensor.distribution.log_prob(value) * advantage


def get_score_function_with_advantage(advantage_fn=None,
                                      name="ScoreFunctionWithAdvantage"):
  """Score function estimator with advantage function.

  Args:
    advantage_fn: callable that takes the `DistributionTensor` and the
      downstream `loss` and returns a `Tensor` advantage
      (e.g. `loss - baseline`).
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `DistributionTensor`, the
    sampled `value`, and the downstream `loss`, and uses the provided advantage.
  """

  def score_function_with_advantage(dist_tensor, value, loss):
    with ops.name_scope(name, values=[value, loss]):
      advantage = advantage_fn(dist_tensor, loss)
      advantage = array_ops.stop_gradient(advantage)
      return dist_tensor.distribution.log_prob(value) * advantage

  return score_function_with_advantage


def get_score_function_with_constant_baseline(baseline, name="ScoreFunction"):
  """Score function estimator with constant baseline.

  Args:
    baseline: `Tensor` to be subtracted from loss.
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `DistributionTensor`, the
    sampled `value`, and the downstream `loss`, and subtracts the provided
    `baseline` from the `loss`.
  """

  def score_function_with_constant_baseline(dist_tensor, value, loss):
    return score_function(dist_tensor, value, loss, baseline, name)

  return score_function_with_constant_baseline


def get_score_function_with_baseline(baseline_fn=None, name="ScoreFunction"):
  """Score function estimator with baseline function.

  Args:
    baseline_fn: callable that takes the `DistributionTensor` and the downstream
      `loss` and returns a `Tensor` baseline to be subtracted from the `loss`.
      If None, defaults to `get_mean_baseline`, which is an EMA of the loss.
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `DistributionTensor`, the
    sampled `value`, and the downstream `loss`, and subtracts the provided
    `baseline` from the `loss`.
  """
  if baseline_fn is None:
    baseline_fn = get_mean_baseline()

  def score_function_with_baseline(dist_tensor, value, loss):
    with ops.name_scope(name):
      b = baseline_fn(dist_tensor, loss)
      return score_function(dist_tensor, value, loss, b)

  return score_function_with_baseline


def get_mean_baseline(ema_decay=0.99, name=None):
  """ExponentialMovingAverage baseline.

  EMA initializes to 0, which introduces a bias. This baseline implements the
  bias correction term from Adam (section 3 of
  https://arxiv.org/pdf/1412.6980v8.pdf), dividing by `1 - ema_decay^t`, where
  `t` is the step count.

  Args:
    ema_decay: decay rate for the ExponentialMovingAverage.
    name: name for variable scope of the ExponentialMovingAverage.

  Returns:
    Callable baseline function that takes the `DistributionTensor` (unused) and
    the downstream `loss`, and returns an EMA of the loss.
  """

  def mean_baseline(_, loss):
    with vs.variable_scope(name, default_name="MeanBaseline"):
      reduced_loss = math_ops.reduce_mean(loss)

      ema = training.ExponentialMovingAverage(decay=ema_decay)
      update_op = ema.apply([reduced_loss])

      # The bias correction term requires keeping track of how many times the
      # EMA has been updated. Creating a variable here to do so. The global step
      # is not used because it may or may not track exactly the number of times
      # the EMA is updated.
      ema_var = ema.average(reduced_loss)
      assert ema_var is not None
      with ops.colocate_with(ema_var):
        num_updates = vs.get_variable(
            "local_ema_step", initializer=0, trainable=False)
      num_updates = num_updates.assign_add(1)
      bias_correction = 1. - math_ops.pow(ema_decay, math_ops.cast(
          num_updates, reduced_loss.dtype))

      with ops.control_dependencies([update_op]):
        baseline = ema.average(reduced_loss) / bias_correction

      return baseline

  return mean_baseline


__all__ = make_all(__name__)
