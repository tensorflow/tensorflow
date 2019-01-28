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
"""Gradients for Popnn operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops

"""
    These gradient function should *never* be called directly.
"""

@ops.RegisterGradient("PopnnLstmLayer")
def _popnn_lstm_layer_backward(op, *grads):
  """Gradients for the PopnnLstmLayer op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use PopnnLstmLayer in gradients, is_training must be set to True."
    )
  return gen_popnn_ops.popnn_lstm_layer_backprop(
      inputs=op.inputs[0],
      input_h_state=op.inputs[1],
      input_c_state=op.inputs[2],
      kernel=op.inputs[3],
      biases=op.inputs[4],
      output=op.outputs[0],
      output_h_state=op.outputs[1],
      output_c_state=op.outputs[2],
      intermediates=op.outputs[3],
      output_backprop=grads[0],
      output_h_state_backprop=grads[1],
      output_c_state_backprop=grads[2],
      num_channels=op.get_attr("num_channels"),
      partials_dtype=op.get_attr("partials_dtype"),
      is_training=op.get_attr("is_training"))


@ops.RegisterGradient("PopnnGroupNormTraining")
def _popnn_group_norm_backward(op, *grads):
  """Gradients for the PopnnGroupNormTraining op."""
  return gen_popnn_ops.popnn_group_norm_grad(
      input_whitened=op.outputs[3],
      gamma=op.inputs[1],
      mean=op.outputs[1],
      inv_std_dev=op.outputs[2],
      output_backprop=grads[0],
      data_format=op.get_attr("data_format"),
      epsilon=op.get_attr("epsilon"),
      num_groups=op.get_attr("num_groups"))
