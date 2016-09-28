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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages


def resample_at_rate(inputs, rates, scope=None, seed=None, back_prop=False):
  """Given `inputs` tensors, stochastically resamples each at a given rate.

  For example, if the inputs are `[[a1, a2], [b1, b2]]` and the rates
  tensor contains `[3, 1]`, then the return value may look like `[[a1,
  a2, a1, a1], [b1, b2, b1, b1]]`. However, many other outputs are
  possible, since this is stochastic -- averaged over many repeated
  calls, each set of inputs should appear in the output `rate` times
  the number of invocations.

  Uses Knuth's method to generate samples from the poisson
  distribution (but instead of just incrementing a count, actually
  emits the input); this is described at
  https://en.wikipedia.org/wiki/Poisson_distribution in the section on
  generating Poisson-distributed random variables.

  Note that this method is not appropriate for large rate values: with
  float16 it will stop performing correctly for rates above 9.17;
  float32, 87; and float64, 708. (These are the base-e versions of the
  minimum representable exponent for each type.)

  Args:
    inputs: A list of tensors, each of which has a shape of `[batch_size, ...]`
    rates: A tensor of shape `[batch_size]` contiaining the resampling rates
           for each input.
    scope: Scope for the op.
    seed: Random seed to use.
    back_prop: Whether to allow back-propagation through this op.

  Returns:
    Selections from the input tensors.

  """
  # TODO(shoutis): Refactor, splitting this up into a poisson draw and a repeat.

  # What this implementation does is loop, simulating the intervals
  # between events by drawing from the exponential distribution
  # (`-log(random_uniform)/rate`), and emitting another copy of the
  # corresponding input so long as sum(intervals) < 1. However, that
  # condition can be transformed into the easier-to-compute condition
  # `product(random_uniforms) > e^-rate`.
  with ops.name_scope(scope, default_name='resample_at_rate', values=inputs):
    floor_vals = math_ops.exp(-rates)

    def _body(chosen_inputs, running_products, idx, output_count):
      """Body of the resampling loop."""
      # Update the running product
      next_running_products = running_products * random_ops.random_uniform(
          shape=array_ops.shape(running_products), seed=seed)

      # Append inputs which still pass the condition:
      indexes = array_ops.reshape(
          array_ops.where(next_running_products > floor_vals), [-1])

      next_output_count = output_count + array_ops.shape(indexes)[0]

      next_chosen_inputs = [
          chosen_inputs[i].write(idx, array_ops.gather(inputs[i], indexes))
          for i in range(len(inputs))]

      return [next_chosen_inputs,
              next_running_products,
              idx + 1,
              next_output_count]

    def _cond(unused_chosen_inputs, running_products, unused_idx, unused_count):
      """Resampling loop exit condition."""
      return math_ops.reduce_any(running_products > floor_vals)

    initial_chosen_inputs = [
        tensor_array_ops.TensorArray(dtype=x.dtype, size=0, dynamic_size=True)
        for x in inputs]

    resampled_inputs, _, unused_idx, count = control_flow_ops.while_loop(
        _cond,
        _body,
        loop_vars=[initial_chosen_inputs,
                   array_ops.ones_like(rates),  # initial running_products
                   0,   # initial idx
                   0],  # initial count
        back_prop=back_prop)

  # Work around TensorArray "Currently only static shapes are supported when
  # concatenating zero-size TensorArrays" limitation:
  def _empty_tensor_like(t):
    result = array_ops.zeros(
        shape=(array_ops.concat(0, [[0], array_ops.shape(t)[1:]])),
        dtype=t.dtype)
    if t.get_shape().ndims is not None:
      # preserve known shapes
      result.set_shape([0] + t.get_shape()[1:].as_list())
    return result

  return control_flow_ops.cond(
      count > 0,
      lambda: [tensor_array.concat() for tensor_array in resampled_inputs],
      lambda: [_empty_tensor_like(t) for t in inputs])


def weighted_resample(inputs, weights, overall_rate, scope=None,
                      mean_decay=0.999, warmup=10, seed=None):
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
    warmup: Until the resulting tensor has been evaluated `warmup`
      times, the resampling menthod uses the true mean over all calls
      as its weight estimate, rather than a decayed mean.
    seed: Random seed.

  Returns:
    A list of tensors exactly like `inputs`, but with an unknown (and
      possibly zero) first dimension.
    A tensor containing the effective resampling rate used for each output.

  """
  # Algorithm: Just compute rates as weights/mean_weight *
  # overall_rate. This way the the average weight corresponds to the
  # overall rate, and a weight twice the average has twice the rate,
  # etc.
  with ops.name_scope(scope, 'weighted_resample', inputs) as opscope:
    # First: Maintain a running estimated mean weight, with decay
    # adjusted (by also maintaining an invocation count) during the
    # warmup period so that at the beginning, there aren't too many
    # zeros mixed in, throwing the average off.

    with variable_scope.variable_scope(scope, 'estimate_mean', inputs):
      count_so_far = variable_scope.get_local_variable(
          'resample_count', initializer=0)

      estimated_mean = variable_scope.get_local_variable(
          'estimated_mean', initializer=0.0)

      count = count_so_far.assign_add(1)
      real_decay = math_ops.minimum(
          math_ops.truediv((count - 1), math_ops.minimum(count, warmup)),
          mean_decay)

      batch_mean = math_ops.reduce_mean(weights)
      mean = moving_averages.assign_moving_average(
          estimated_mean, batch_mean, real_decay)

    # Then, normalize the weights into rates using the mean weight and
    # overall target rate:
    rates = weights * overall_rate / mean

    results = resample_at_rate([rates] + inputs, rates,
                               scope=opscope, seed=seed, back_prop=False)

    return (results[1:], results[0])
