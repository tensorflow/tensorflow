# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Generating TF functions which can be 1:1 mapped to TFLite ops."""

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util.compat import as_bytes

tfe = tf.contrib.eager


def _annotate_tflite_op(outputs, op_name, options=None):
  """Annotates a TF Node with TFLite options.

  Args:
    outputs: The output(s) of the op. Can be a `Tensor` or a list of `Tensor`s.
    op_name: The TensorFlow Lite Op name. E.g. 'FULLY_CONNECTED'.
    options: A `dict` which contains TensorFlow Lite options.

  Raises:
    ValueError: If unsupported option types are used.
  """

  # The outputs may be a `Tensor` or a list of tensors.
  if isinstance(outputs, tf.Tensor):
    op = outputs.op
  else:
    op = outputs[0].op

  # pylint: disable=protected-access
  # Note: `as_bytes` conversion is required for Python 3.
  op._set_attr(
      '_tflite_function_name', attr_value_pb2.AttrValue(s=as_bytes(op_name)))
  if options:
    for key, value in options.items():
      if isinstance(value, str):
        op._set_attr(key, attr_value_pb2.AttrValue(s=as_bytes(value)))
      elif isinstance(value, int):
        op._set_attr(key, attr_value_pb2.AttrValue(i=value))
      else:
        raise ValueError('Unsupported option value type %s' % value.__class__)
  # pylint: enable=protected-access


# TODO(ycling): Generate this interface with FlatBuffer reflection
# functionality and extend the coverage to all TFLiteops.
# TODO(ycling): Support optional tensors (e.g. not using missing bias).
def fully_connected(x, weights, bias, fused_activation_function='NONE'):
  """Create a TF function node equalivent to TFLite FULLY_CONNECTED op."""
  options = {'_fused_activation_function': fused_activation_function}

  @tfe.defun
  def tf_lite_fully_connected(x, weights, bias):
    """The TFLite FULLY_CONNECTED logic wrapped in a TF Function."""
    # TFLite FULLY_CONNECTED definition is different from TF matmul.
    # The weights are transposed. Therefore we need to transpose the
    # weights inside the TF function to simulate TFLite behavior.
    transposed_weights = tf.transpose(weights)

    y = tf.matmul(x, transposed_weights) + bias
    if fused_activation_function == 'RELU':
      y = tf.nn.relu(y)
    elif fused_activation_function == 'NONE':
      # Do nothing.
      pass
    else:
      # TODO(ycling): Support other activation functions.
      raise Exception('Unsupported fused_activation_function "%s"' %
                      fused_activation_function)
    return y

  output = tf_lite_fully_connected(x, weights, bias)
  _annotate_tflite_op(output, 'FULLY_CONNECTED', options=options)

  return output
