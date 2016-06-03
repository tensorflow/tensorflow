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
"""Quantized Math Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.quantization.ops import gen_math_ops
from tensorflow.contrib.quantization.ops.gen_math_ops import *
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import common_shapes


# QuantizedMatMul* ops.
@ops.RegisterShape("QuantizedMatMul")
def _QuantizedMatMulShape(op):
  unused_a_min = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  unused_a_max = op.inputs[3].get_shape().merge_with(tensor_shape.scalar())
  unused_b_min = op.inputs[4].get_shape().merge_with(tensor_shape.scalar())
  unused_b_max = op.inputs[5].get_shape().merge_with(tensor_shape.scalar())
  result = common_shapes.matmul_shape(op)
  result.extend([tensor_shape.scalar(), tensor_shape.scalar()])
  return result
