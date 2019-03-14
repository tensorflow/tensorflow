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

  Call Arguments:

    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
        given, will use `value` for both `key` and `value`, which is the
        most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.

  Output shape:

    Attention outputs of shape `[batch_size, Tq, dim]`.
  """

  def _calculate_scores(self, query, key):
    """Calculates attention scores.

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    return NotImplementedError

  def _apply_scores(self, scores, value, value_mask=None):
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

  # TODO(b/125916026): Consider exposing a __call__ method with named args.
  def call(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    # TODO(b/125916026): Support query_mask.
    if q_mask is not None:
      raise NotImplementedError('query_mask is not supported yet.')
    scores = self._calculate_scores(query=q, key=k)
    return self._apply_scores(scores=scores, value=v, value_mask=v_mask)

  def _validate_call_args(self, inputs, mask):
    """Validates arguments of the call method."""
    class_name = self.__class__.__name__
    if not isinstance(inputs, list):
      raise ValueError(
          '{} layer must be called on a list of inputs, namely [query, value] '
          'or [query, value, key].'.format(class_name))
    if len(inputs) < 2 or len(inputs) > 3:
      raise ValueError(
          '{} layer accepts inputs list of length 2 or 3, '
          'namely [query, value] or [query, value, key]. '
          'Given length: {}'.format(class_name, len(inputs)))
    if mask:
      if not isinstance(mask, list):
        raise ValueError(
            '{} layer mask must be a list, '
            'namely [query_mask, value_mask].'.format(class_name))
      if len(mask) != 2:
        raise ValueError(
            '{} layer mask must be a list of length 2, namely [query_mask, '
            'value_mask]. Given length: {}'.format(class_name, len(mask)))


class Attention(BaseDenseAttention):
  """Dot-product attention layer, a.k.a. Luong-style attention.

  Inputs are `query` tensor of shape `[batch_size, Tq]`, `value` tensor of shape
  `[batch_size, Tv]` and `key` tensor of shape `[batch_size, Tv]`.
  The calculation follows the steps:

  1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
     product: `scores = tf.matmul(query, key, transpose_b=True)`.
  2. Use scores to calculate a distribution with shape
     `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
  3. Use `distribution` to create a linear combination of `value` with
     shape `batch_size, Tq, dim]`:
     `return tf.matmul(distribution, value)`.

  Args:
    scale: If `True`, will create a scalar variable to scale the attention
      scores.

  Call Arguments:

    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
        given, will use `value` for both `key` and `value`, which is the
        most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.

  Output shape:

    Attention outputs of shape `[batch_size, Tq, dim]`.

  The meaning of `query`, `value` and `key` depend on the application. In the
  case of text similarity, for example, `query` is the sequence embeddings of
  the first piece of text and `value` is the sequence embeddings of the second
  piece of text. `key` is usually the same tensor as `value`.

  Here is a code example for using `Attention` in a CNN+Attention network:

  ```python
  # Variable-length int sequences.
  query_input = tf.keras.Input(shape=(None,), dtype='int32')
  value_input = tf.keras.Input(shape=(None,), dtype='int32')

  # Embedding lookup.
  token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
  # Query embeddings of shape [batch_size, Tq, dimension].
  query_embeddings = token_embedding(query_input)
  # Value embeddings of shape [batch_size, Tv, dimension].
  value_embeddings = token_embedding(query_input)

  # CNN layer.
  cnn_layer = tf.keras.layers.Conv1D(
      filters=100,
      kernel_size=4,
      # Use 'same' padding so outputs have the same shape as inputs.
      padding='same')
  # Query encoding of shape [batch_size, Tq, filters].
  query_seq_encoding = cnn_layer(query_embeddings)
  # Value encoding of shape [batch_size, Tv, filters].
  value_seq_encoding = cnn_layer(value_embeddings)

  # Query-value attention of shape [batch_size, Tq, filters].
  query_value_attention_seq = tf.keras.layers.Attention()(
      [query_seq_encoding, value_seq_encoding])

  # Reduce over the sequence axis to produce encodings of shape
  # [batch_size, filters].
  query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
      query_seq_encoding)
  query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
      query_value_attention_seq)

  # Concatenate query and document encodings to produce a DNN input layer.
  input_layer = tf.keras.layers.Concatenate()(
      [query_encoding, query_value_attention])

  # Add DNN layers, and create Model.
  # ...
  ```
  """

  def __init__(self, scale=False, **kwargs):
    super(Attention, self).__init__(**kwargs)
    # TODO(b/125916026): Support scale.
    if scale:
      raise NotImplementedError('scale=True is not supported yet.')
    self.scale = scale

  def build(self, input_shape):
    """Creates scale variable if scale==True."""
    # TODO(b/125916026): Create scale variable if self.scale is True.
    self.scale_var = None
    super(Attention, self).build(input_shape)

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a query-key dot product.

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    scores = math_ops.matmul(query, key, transpose_b=True)
    if self.scale_var is not None:
      scores *= self.scale_var
    return scores
