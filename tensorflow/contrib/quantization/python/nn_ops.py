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
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.quantization.ops import gen_nn_ops
from tensorflow.contrib.quantization.ops.gen_nn_ops import *
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import common_shapes


# QuantizedAvgPool* ops.
@ops.RegisterShape("QuantizedAvgPool")
def _QuantizedAvgPoolShape(op):
  return [common_shapes.avg_pool_shape(op)[0], tensor_shape.scalar(),
          tensor_shape.scalar()]


# QuantizedBiasAdd op.
@ops.RegisterShape("QuantizedBiasAdd")
def _QuantizedBiasAddShape(op):
  """Returns the same shape as the input, plus min and max scalar values.

  Args:
    op: Input operation.
  Returns:
    Shape of ops first input, plus min and max tensors.
  """
  unused_input_min = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  unused_input_max = op.inputs[3].get_shape().merge_with(tensor_shape.scalar())
  unused_bias_min = op.inputs[4].get_shape().merge_with(tensor_shape.scalar())
  unused_bias_max = op.inputs[5].get_shape().merge_with(tensor_shape.scalar())
  return [op.inputs[0].get_shape(), tensor_shape.scalar(),
          tensor_shape.scalar()]


# QuantizedConv2D* ops.
@ops.RegisterShape("QuantizedConv2D")
def _QuantizedConv2DShape(op):
  """Returns the same shape as Conv2D, plus min and max scalar values.

  Args:
    op: Input operation.
  Returns:
    Shape of float Conv2D, plus min and max tensors.
  """
  unused_input_min = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  unused_input_max = op.inputs[3].get_shape().merge_with(tensor_shape.scalar())
  unused_filter_min = op.inputs[4].get_shape().merge_with(tensor_shape.scalar())
  unused_filter_max = op.inputs[5].get_shape().merge_with(tensor_shape.scalar())
  result = common_shapes.conv2d_shape(op)
  result.extend([tensor_shape.scalar(), tensor_shape.scalar()])
  return result


# QuantizedMaxPool* ops.
@ops.RegisterShape("QuantizedMaxPool")
def _QuantizedMaxPoolShape(op):
  return [common_shapes.max_pool_shape(op)[0], tensor_shape.scalar(),
          tensor_shape.scalar()]


@ops.RegisterShape("QuantizedRelu")
@ops.RegisterShape("QuantizedRelu6")
@ops.RegisterShape("QuantizedReluX")
@ops.RegisterShape("QuantizeDownAndShrinkRange")
def _QuantizedSameShape(op):
  unused_min = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  unused_max = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  return [op.inputs[0].get_shape(), tensor_shape.scalar(),
          tensor_shape.scalar()]
