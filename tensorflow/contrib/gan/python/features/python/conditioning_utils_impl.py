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
"""Miscellanous utilities for TFGAN code and examples.

Includes:
1) Conditioning the value of a Tensor, based on techniques from
  https://arxiv.org/abs/1609.03499.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


__all__ = [
    'condition_tensor',
    'condition_tensor_from_onehot',
]


def _get_shape(tensor):
  tensor_shape = array_ops.shape(tensor)
  static_tensor_shape = tensor_util.constant_value(tensor_shape)
  return (static_tensor_shape if static_tensor_shape is not None else
          tensor_shape)


def condition_tensor(tensor, conditioning):
  """Condition the value of a tensor.

  Conditioning scheme based on https://arxiv.org/abs/1609.03499.

  Args:
    tensor: A minibatch tensor to be conditioned.
    conditioning: A minibatch Tensor of to condition on. Must be 2D, with first
      dimension the same as `tensor`.

  Returns:
    `tensor` conditioned on `conditioning`.

  Raises:
    ValueError: If the non-batch dimensions of `tensor` aren't fully defined.
    ValueError: If `conditioning` isn't at least 2D.
    ValueError: If the batch dimension for the input Tensors don't match.
  """
  tensor.shape[1:].assert_is_fully_defined()
  num_features = tensor.shape[1:].num_elements()

  mapped_conditioning = layers.linear(
      layers.flatten(conditioning), num_features)
  if not mapped_conditioning.shape.is_compatible_with(tensor.shape):
    mapped_conditioning = array_ops.reshape(
        mapped_conditioning, _get_shape(tensor))
  return tensor + mapped_conditioning


def _one_hot_to_embedding(one_hot, embedding_size):
  """Get a dense embedding vector from a one-hot encoding."""
  num_tokens = one_hot.shape[1]
  label_id = math_ops.argmax(one_hot, axis=1)
  embedding = variable_scope.get_variable(
      'embedding', [num_tokens, embedding_size])
  return embedding_ops.embedding_lookup(
      embedding, label_id, name='token_to_embedding')


def _validate_onehot(one_hot_labels):
  one_hot_labels.shape.assert_has_rank(2)
  one_hot_labels.shape[1:].assert_is_fully_defined()


def condition_tensor_from_onehot(tensor, one_hot_labels, embedding_size=256):
  """Condition a tensor based on a one-hot tensor.

  Conditioning scheme based on https://arxiv.org/abs/1609.03499.

  Args:
    tensor: Tensor to be conditioned.
    one_hot_labels: A Tensor of one-hot labels. Shape is
      [batch_size, num_classes].
    embedding_size: The size of the class embedding.

  Returns:
    `tensor` conditioned on `one_hot_labels`.

  Raises:
    ValueError: `one_hot_labels` isn't 2D, if non-batch dimensions aren't
      fully defined, or if batch sizes don't match.
  """
  _validate_onehot(one_hot_labels)

  conditioning = _one_hot_to_embedding(one_hot_labels, embedding_size)
  return condition_tensor(tensor, conditioning)
