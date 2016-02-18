# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Operations for implementing attention.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_attention_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_attention_ops import *
# pylint: enable=wildcard-import


# TODO(bsteiner): Implement the gradient function for extract_glimpse
ops.NoGradient("ExtractGlimpse")


@ops.RegisterShape("ExtractGlimpse")
def _ExtractGlimpseShape(op):
  """Shape function for ExtractGlimpse op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  unused_size_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(2))
  offsets_shape = op.inputs[2].get_shape().merge_with(
      input_shape[:1].concatenate([2]))
  offsets_shape = offsets_shape
  size_value = tensor_util.constant_value(op.inputs[1])
  if size_value is not None:
    height = size_value[0]
    width = size_value[1]
  else:
    height = None
    width = None
  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, input_shape[3]])]
