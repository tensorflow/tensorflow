# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Utils related to keras model saving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest


def extract_model_metrics(model):
  """Convert metrics from a Keras model to (value, update) ops.

  This is used for converting Keras models to Estimators and SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to tuples of (value, update) ops. May return
    `None` if the model does not contain any metrics.
  """
  from tensorflow.python.keras import metrics  # pylint: disable=g-import-not-at-top
  if not getattr(model, '_compile_metrics', None):
    return None

  # TODO(psv/kathywu): use this implementation in model to estimator flow.
  eval_metric_ops = {}
  for metric_name in model.metrics_names[1:]:  # Index 0 is `loss`.
    m = metrics.Mean()
    m(model._compile_metrics_tensors[metric_name])
    eval_metric_ops[metric_name] = m
  return eval_metric_ops


def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature

  if input_signature is None:
    try:
      inputs = model.inputs
      input_names = model.input_names
    except AttributeError:
      raise ValueError(
          'Model {} cannot be saved because the input shapes have not been '
          'set. Usually, input shapes are automatically determined from calling'
          ' .fit() or .predict(). To manually set the shapes, call '
          'model._set_inputs(inputs).'.format(model))
    input_specs = []
    for input_tensor, input_name in zip(inputs, input_names):
      input_specs.append(tensor_spec.TensorSpec(
          shape=input_tensor.shape, dtype=input_tensor.dtype,
          name=input_name))
    # The input signature of the call function is a list with one element, since
    # all tensor inputs must be passed in as the first argument.
    input_signature = [input_specs] if len(input_specs) > 1 else input_specs

  # TODO(mdan): Should the model's call be autographed by default?
  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)
    outputs_list = nest.flatten(model(inputs=inputs))
    try:
      output_names = model.output_names
    except AttributeError:
      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
      output_names = training_utils.generic_output_names(outputs_list)
    return {name: output for name, output in zip(output_names, outputs_list)}

  return _wrapped_model
