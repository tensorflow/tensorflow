# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Attention layers that can be used in sequence DNN/CNN models.

This file follows the terminology of https://arxiv.org/abs/1706.03762 Figure 2.
Attention is formed by three tensors: Query, Key and Value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class BaseDenseAttention(Layer):
  """Base Attention class for Dense networks.

  This class is suitable for Dense or CNN networks, and not for RNN networks.

  Implementations of attention mechanisms should inherit from this class, and
  reuse the `apply_attention_scores()` method.
  """

  def apply_attention_scores(self, scores, value, value_mask=None):
    """Applies attention scores to the given value tensor.

    To use this method in your attention layer, follow the steps:

    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `value_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

    Args:
      scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
      value: Value tensor of shape `[batch_size, Tv, dim]`.
      value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.

    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.
    """
    if value_mask is not None:
      # Mask of shape [batch_size, 1, Tv] that is True in padding position.
      padding_mask = array_ops.expand_dims(
          math_ops.logical_not(value_mask), axis=1)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
    attention_distribution = nn.softmax(scores)
    return math_ops.matmul(attention_distribution, value)
