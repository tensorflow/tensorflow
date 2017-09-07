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

These functions are meant to be used in conjunction with `StochasticTensor`
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

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import training
from tensorflow.python.util.all_util import make_all


def score_function(stochastic_tensor, value, loss, baseline=None,
                   name="ScoreFunction"):
  """Score function estimator.

  Computes the integrand of the score function with a baseline:
  `p.log_prob(value) * (loss - baseline)`.

  It will add a `stop_gradient` to the advantage `(loss - baseline)`.

  Args:
    stochastic_tensor: `StochasticTensor` p(x).
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
    return stochastic_tensor.distribution.log_prob(value) * advantage


def get_score_function_with_advantage(advantage_fn=None,
                                      name="ScoreFunctionWithAdvantage"):
  """Score function estimator with advantage function.

  Args:
    advantage_fn: callable that takes the `StochasticTensor` and the
      downstream `loss` and returns a `Tensor` advantage
      (e.g. `loss - baseline`).
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `StochasticTensor`, the
    sampled `value`, and the downstream `loss`, and uses the provided advantage.
  """

  def score_function_with_advantage(stochastic_tensor, value, loss):
    with ops.name_scope(name, values=[value, loss]):
      advantage = advantage_fn(stochastic_tensor, loss)
      advantage = array_ops.stop_gradient(advantage)
      return stochastic_tensor.distribution.log_prob(value) * advantage

  return score_function_with_advantage


def get_score_function_with_constant_baseline(baseline, name="ScoreFunction"):
  """Score function estimator with constant baseline.

  Args:
    baseline: `Tensor` to be subtracted from loss.
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `StochasticTensor`, the
    sampled `value`, and the downstream `loss`, and subtracts the provided
    `baseline` from the `loss`.
  """

  def score_function_with_constant_baseline(stochastic_tensor, value, loss):
    return score_function(stochastic_tensor, value, loss, baseline, name)

  return score_function_with_constant_baseline


def get_score_function_with_baseline(baseline_fn=None, name="ScoreFunction"):
  """Score function estimator with baseline function.

  Args:
    baseline_fn: callable that takes the `StochasticTensor` and the downstream
      `loss` and returns a `Tensor` baseline to be subtracted from the `loss`.
      If None, defaults to `get_mean_baseline`, which is an EMA of the loss.
    name: name to prepend ops with.

  Returns:
    Callable score function estimator that takes the `StochasticTensor`, the
    sampled `value`, and the downstream `loss`, and subtracts the provided
    `baseline` from the `loss`.
  """
  if baseline_fn is None:
    baseline_fn = get_mean_baseline()

  def score_function_with_baseline(stochastic_tensor, value, loss):
    with ops.name_scope(name):
      b = baseline_fn(stochastic_tensor, loss)
      return score_function(stochastic_tensor, value, loss, b)

  return score_function_with_baseline


def get_mean_baseline(ema_decay=0.99, name=None):
  """ExponentialMovingAverage baseline.

  Args:
    ema_decay: decay rate for the ExponentialMovingAverage.
    name: name for variable scope of the ExponentialMovingAverage.

  Returns:
    Callable baseline function that takes the `StochasticTensor` (unused) and
    the downstream `loss`, and returns an EMA of the loss.
  """

  def mean_baseline(_, loss):
    with vs.variable_scope(name, default_name="MeanBaseline"):
      reduced_loss = math_ops.reduce_mean(loss)

      ema = training.ExponentialMovingAverage(decay=ema_decay, zero_debias=True)
      update_op = ema.apply([reduced_loss])

      with ops.control_dependencies([update_op]):
        # Using `identity` causes an op to be added in this context, which
        # triggers the update. Removing the `identity` means nothing is updated.
        baseline = array_ops.identity(ema.average(reduced_loss))

      return baseline

  return mean_baseline


def get_vimco_advantage_fn(have_log_loss=False):
  """VIMCO (Variational Inference for Monte Carlo Objectives) baseline.

  Implements VIMCO baseline from the article of the same name:

  https://arxiv.org/pdf/1602.06725v2.pdf

  Given a `loss` tensor (containing non-negative probabilities or ratios),
  calculates the advantage VIMCO advantage via Eq. 9 of the above paper.

  The tensor `loss` should be shaped `[n, ...]`, with rank at least 1.  Here,
  the first axis is considered the single sampling dimension and `n` must
  be at least 2.  Specifically, the `StochasticTensor` is assumed to have
  used the `SampleValue(n)` value type with `n > 1`.

  Args:
    have_log_loss: Python `Boolean`.  If `True`, the loss is assumed to be the
      log loss.  If `False` (the default), it is assumed to be a nonnegative
      probability or probability ratio.

  Returns:
    Callable baseline function that takes the `StochasticTensor` (unused) and
    the downstream `loss`, and returns the VIMCO baseline for the loss.
  """
  def vimco_advantage_fn(_, loss, name=None):
    """Internal VIMCO function.

    Args:
      _: ignored `StochasticTensor`.
      loss: The loss `Tensor`.
      name: Python string, the name scope to use.

    Returns:
      The advantage `Tensor`.
    """
    with ops.name_scope(name, "VIMCOAdvantage", values=[loss]):
      loss = ops.convert_to_tensor(loss)
      loss_shape = loss.get_shape()
      loss_num_elements = loss_shape[0].value
      n = math_ops.cast(
          loss_num_elements or array_ops.shape(loss)[0], dtype=loss.dtype)

      if have_log_loss:
        log_loss = loss
      else:
        log_loss = math_ops.log(loss)

      # Calculate L_hat, Eq. (4) -- stably
      log_mean = math_ops.reduce_logsumexp(log_loss, [0]) - math_ops.log(n)

      # expand_dims: Expand shape [a, b, c] to [a, 1, b, c]
      log_loss_expanded = array_ops.expand_dims(log_loss, [1])

      # divide: log_loss_sub with shape [a, a, b, c], where
      #
      #  log_loss_sub[i] = log_loss - log_loss[i]
      #
      #       = [ log_loss[j] - log_loss[i] for rows j = 0 ... i - 1     ]
      #         [ zeros                                                  ]
      #         [ log_loss[j] - log_loss[i] for rows j = i + 1 ... a - 1 ]
      #
      log_loss_sub = log_loss - log_loss_expanded

      # reduce_sum: Sums each row across all the sub[i]'s; result is:
      #   reduce_sum[j] = (n - 1) * log_loss[j] - (sum_{i != j} loss[i])
      # divide by (n - 1) to get:
      #   geometric_reduction[j] =
      #     log_loss[j] - (sum_{i != j} log_loss[i]) / (n - 1)
      geometric_reduction = math_ops.reduce_sum(log_loss_sub, [0]) / (n - 1)

      # subtract this from the original log_loss to get the baseline:
      #   geometric_mean[j] = exp((sum_{i != j} log_loss[i]) / (n - 1))
      log_geometric_mean = log_loss - geometric_reduction

      ## Equation (9)

      # Calculate sum_{i != j} loss[i] -- via exp(reduce_logsumexp(.))
      # reduce_logsumexp: log-sum-exp each row across all the
      # -sub[i]'s, result is:
      #
      #  exp(reduce_logsumexp[j]) =
      #    1 + sum_{i != j} exp(log_loss[i] - log_loss[j])
      log_local_learning_reduction = math_ops.reduce_logsumexp(
          -log_loss_sub, [0])

      # convert local_learning_reduction to the sum-exp of the log-sum-exp
      #  (local_learning_reduction[j] - 1) * exp(log_loss[j])
      #    = sum_{i != j} exp(log_loss[i])
      local_learning_log_sum = (
          _logexpm1(log_local_learning_reduction) + log_loss)

      # Add (logaddexp) the local learning signals (Eq. 9)
      local_learning_signal = (
          math_ops.reduce_logsumexp(
              array_ops.stack((local_learning_log_sum, log_geometric_mean)),
              [0])
          - math_ops.log(n))

      advantage = log_mean - local_learning_signal

      return advantage

  return vimco_advantage_fn


def _logexpm1(x):
  """Stably calculate log(exp(x)-1)."""
  with ops.name_scope("logsumexp1"):
    eps = np.finfo(x.dtype.as_numpy_dtype).eps
    # Choose a small offset that makes gradient calculations stable for
    # float16, float32, and float64.
    safe_log = lambda y: math_ops.log(y + eps / 1e8)  # For gradient stability
    return array_ops.where(
        math_ops.abs(x) < eps,
        safe_log(x) + x/2 + x*x/24,  # small x approximation to log(expm1(x))
        safe_log(math_ops.exp(x) - 1))


__all__ = make_all(__name__)
