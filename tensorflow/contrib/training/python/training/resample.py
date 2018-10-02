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

"""Resampling methods for batches of tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages


def _repeat_range(counts, name=None):
  """Repeat integers given by range(len(counts)) each the given number of times.

  Example behavior:
  [0, 1, 2, 3] -> [1, 2, 2, 3, 3, 3]

  Args:
    counts: 1D tensor with dtype=int32.
    name: optional name for operation.

  Returns:
    1D tensor with dtype=int32 and dynamic length giving the repeated integers.
  """
  with ops.name_scope(name, 'repeat_range', [counts]) as scope:
    counts = ops.convert_to_tensor(counts, name='counts')

    def cond(unused_output, i):
      return i < size

    def body(output, i):
      value = array_ops.fill(counts[i:i+1], i)
      return (output.write(i, value), i + 1)

    size = array_ops.shape(counts)[0]
    init_output_array = tensor_array_ops.TensorArray(
        dtype=dtypes.int32, size=size, infer_shape=False)
    output_array, num_writes = control_flow_ops.while_loop(
        cond, body, loop_vars=[init_output_array, 0])

    return control_flow_ops.cond(
        num_writes > 0,
        output_array.concat,
        lambda: array_ops.zeros(shape=[0], dtype=dtypes.int32),
        name=scope)


def resample_at_rate(inputs, rates, scope=None, seed=None, back_prop=False):
  """Given `inputs` tensors, stochastically resamples each at a given rate.

  For example, if the inputs are `[[a1, a2], [b1, b2]]` and the rates
  tensor contains `[3, 1]`, then the return value may look like `[[a1,
  a2, a1, a1], [b1, b2, b1, b1]]`. However, many other outputs are
  possible, since this is stochastic -- averaged over many repeated
  calls, each set of inputs should appear in the output `rate` times
  the number of invocations.

  Args:
    inputs: A list of tensors, each of which has a shape of `[batch_size, ...]`
    rates: A tensor of shape `[batch_size]` containing the resampling rates
       for each input.
    scope: Scope for the op.
    seed: Random seed to use.
    back_prop: Whether to allow back-propagation through this op.

  Returns:
    Selections from the input tensors.
  """
  with ops.name_scope(scope, default_name='resample_at_rate',
                      values=list(inputs) + [rates]):
    rates = ops.convert_to_tensor(rates, name='rates')
    sample_counts = math_ops.cast(
        random_ops.random_poisson(rates, (), rates.dtype, seed=seed),
        dtypes.int32)
    sample_indices = _repeat_range(sample_counts)
    if not back_prop:
      sample_indices = array_ops.stop_gradient(sample_indices)
    return [array_ops.gather(x, sample_indices) for x in inputs]


def weighted_resample(inputs, weights, overall_rate, scope=None,
                      mean_decay=0.999, seed=None):
  """Performs an approximate weighted resampling of `inputs`.

  This method chooses elements from `inputs` where each item's rate of
  selection is proportional to its value in `weights`, and the average
  rate of selection across all inputs (and many invocations!) is
  `overall_rate`.

  Args:
    inputs: A list of tensors whose first dimension is `batch_size`.
    weights: A `[batch_size]`-shaped tensor with each batch member's weight.
    overall_rate: Desired overall rate of resampling.
    scope: Scope to use for the op.
    mean_decay: How quickly to decay the running estimate of the mean weight.
    seed: Random seed.

  Returns:
    A list of tensors exactly like `inputs`, but with an unknown (and
      possibly zero) first dimension.
    A tensor containing the effective resampling rate used for each output.
  """
  # Algorithm: Just compute rates as weights/mean_weight *
  # overall_rate. This way the average weight corresponds to the
  # overall rate, and a weight twice the average has twice the rate,
  # etc.
  with ops.name_scope(scope, 'weighted_resample', inputs) as opscope:
    # First: Maintain a running estimated mean weight, with zero debiasing
    # enabled (by default) to avoid throwing the average off.

    with variable_scope.variable_scope(scope, 'estimate_mean', inputs):
      estimated_mean = variable_scope.get_local_variable(
          'estimated_mean',
          initializer=math_ops.cast(0, weights.dtype),
          dtype=weights.dtype)

      batch_mean = math_ops.reduce_mean(weights)
      mean = moving_averages.assign_moving_average(
          estimated_mean, batch_mean, mean_decay)

    # Then, normalize the weights into rates using the mean weight and
    # overall target rate:
    rates = weights * overall_rate / mean

    results = resample_at_rate([rates] + inputs, rates,
                               scope=opscope, seed=seed, back_prop=False)

    return (results[1:], results[0])
