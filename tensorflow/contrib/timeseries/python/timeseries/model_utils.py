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
"""Helper functions for training and constructing time series Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries import feature_keys

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


# TODO(agarwal): Remove and replace with functionality from tf.slim
def fully_connected(inp,
                    inp_size,
                    layer_size,
                    name,
                    activation=nn_ops.relu,
                    dtype=dtypes.float32):
  """Helper method to create a fully connected hidden layer."""
  wt = variable_scope.get_variable(
      name="{}_weight".format(name), shape=[inp_size, layer_size], dtype=dtype)
  bias = variable_scope.get_variable(
      name="{}_bias".format(name),
      shape=[layer_size],
      initializer=init_ops.zeros_initializer())
  output = nn_ops.xw_plus_b(inp, wt, bias)
  if activation is not None:
    assert callable(activation)
    output = activation(output)
  return output


def parameter_switch(parameter_overrides):
  """Create a function which chooses between overridden and model parameters.

  Args:
    parameter_overrides: A dictionary with explicit overrides of model
        parameters, mapping from Tensors to their overridden values.
  Returns:
    A function which takes a Tensor and returns the override if it is specified,
        or otherwise the evaluated value (given current Variable values).
  """
  def get_passed_or_trained_value(parameter):
    return ops.convert_to_tensor(
        parameter_overrides.get(parameter, parameter)).eval()
  return get_passed_or_trained_value


def canonicalize_times_or_steps_from_output(times, steps,
                                            previous_model_output):
  """Canonicalizes either relative or absolute times, with error checking."""
  if steps is not None and times is not None:
    raise ValueError("Only one of `steps` and `times` may be specified.")
  if steps is None and times is None:
    raise ValueError("One of `steps` and `times` must be specified.")
  if times is not None:
    times = numpy.array(times)
    if len(times.shape) != 2:
      times = times[None, ...]
    if (previous_model_output[feature_keys.FilteringResults.TIMES].shape[0] !=
        times.shape[0]):
      raise ValueError(
          ("`times` must have a batch dimension matching"
           " the previous model output (got a batch dimension of {} for `times`"
           " and {} for the previous model output).").format(
               times.shape[0], previous_model_output[
                   feature_keys.FilteringResults.TIMES].shape[0]))
    if not (previous_model_output[feature_keys.FilteringResults.TIMES][:, -1] <
            times[:, 0]).all():
      raise ValueError("Prediction times must be after the corresponding "
                       "previous model output.")
  if steps is not None:
    predict_times = (
        previous_model_output[feature_keys.FilteringResults.TIMES][:, -1:] + 1 +
        numpy.arange(steps)[None, ...])
  else:
    predict_times = times
  return predict_times
