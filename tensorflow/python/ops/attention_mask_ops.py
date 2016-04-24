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

"""## Masking

TensorFlow provides several operations that can help you mask the energies of
an attention model.

@@attention_mask
@@attention_mask_median

"""
import sys
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import common_shapes
# pylint: disable=wildcard-import
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.ops.constant_op import constant
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import gen_attention_mask_ops


def attention_mask(sequence_len, input, name=None):
  """Apply mask based on sequence length.

  In general, you want to apply this on the attention energies pre-normalization
  (i.e., softmax).

  Args:
    sequence_len: A `Tensor` of type `int64`.
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_attention_mask_ops._attention_mask(
      sequence_len=sequence_len, input=input,
      fill_value=-np.finfo(np.float32).max, name=name)

@ops.RegisterShape("AttentionMask")
def _AttentionMaskShape(op):
  return [op.inputs[1].get_shape()]

@ops.RegisterGradient("AttentionMask")
def _AttentionMaskGrad(op, *grad):
  attention_mask_grad = gen_attention_mask_ops._attention_mask(
      sequence_len=op.inputs[0], input=grad[0], fill_value=0.0)
  return [None] + [attention_mask_grad]

def attention_mask_median(sequence_len, input, prev_alignment, window_l=None,
                          window_r=None, name=None):
  """Apply mask based on sequence length and median of previous attention.

  In general, you want to apply this on the attention energies pre-normalization
  (i.e., softmax). See Chorowski et al., "Attention-Based Models for Speech
  Recognition", 2015.

  Args:
    sequence_len: A `Tensor` of type `int64`.
    input: A `Tensor` of type `float32`.
    prev_alignment: A `Tensor` of type `float32`.
    window_l: An optional `int`. Defaults to `10`.
    window_r: An optional `int`. Defaults to `200`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_attention_mask_ops._attention_mask_median(
      sequence_len=sequence_len, input=input,
      prev_alignment=prev_alignment, fill_value=-np.finfo(np.float32).max,
      window_l=window_l, window_r=window_r, name=name)

@ops.RegisterShape("AttentionMaskMedian")
def _AttentionMaskMedianShape(op):
  return [op.inputs[1].get_shape()]

@ops.RegisterGradient("AttentionMaskMedian")
def _AttentionMaskMedianGrad(op, *grad):
  attention_mask_grad = gen_attention_mask_ops._attention_mask_median(
      sequence_len=op.inputs[0], input=grad[0], prev_alignment=op.inputs[2],
      fill_value=0.0, window_l=op.get_attr("window_l"),
      window_r=op.get_attr("window_r"))
  return [None] * 2 + [attention_mask_grad]
