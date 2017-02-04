# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Miscellaneous functions useful for nD-LSTM models.

Some of these functions duplicate functionality in tfslim with
slightly different interfaces.

Tensors in this library generally have the shape (num_images, height, width,
depth).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def _shape(tensor):
  """Get the shape of a tensor as an int list."""
  return tensor.get_shape().as_list()


def pixels_as_vector(images, scope=None):
  """Reduce images to vectors by combining all pixels."""
  with ops.name_scope(scope, "PixelsAsVector", [images]):
    batch_size, height, width, depth = _shape(images)
    return array_ops.reshape(images, [batch_size, height * width * depth])


def pool_as_vector(images, scope=None):
  """Reduce images to vectors by averaging all pixels."""
  with ops.name_scope(scope, "PoolAsVector", [images]):
    return math_ops.reduce_mean(images, [1, 2])


def one_hot_planes(labels, num_classes, scope=None):
  """Compute 1-hot encodings for planes.

  Given a label, this computes a label image that contains
  1 at all pixels in the plane corresponding to the target
  class and 0 in all other planes.

  Args:
    labels: (batch_size,) tensor
    num_classes: number of classes
    scope: optional scope name

  Returns:
    Tensor of shape (batch_size, 1, 1, num_classes) with a 1-hot encoding.
  """
  with ops.name_scope(scope, "OneHotPlanes", [labels]):
    batch_size, = _shape(labels)
    batched = layers.one_hot_encoding(labels, num_classes)
    return array_ops.reshape(batched, [batch_size, 1, 1, num_classes])


def one_hot_mask(labels, num_classes, scope=None):
  """Compute 1-hot encodings for masks.

  Given a label image, this computes the one hot encoding at
  each pixel.

  Args:
    labels: (batch_size, width, height, 1) tensor containing labels.
    num_classes: number of classes
    scope: optional scope name

  Returns:
    Tensor of shape (batch_size, width, height, num_classes) with
    a 1-hot encoding.
  """
  with ops.name_scope(scope, "OneHotMask", [labels]):
    height, width, depth = _shape(labels)
    assert depth == 1
    sparse_labels = math_ops.to_int32(array_ops.reshape(labels, [-1, 1]))
    sparse_size, _ = _shape(sparse_labels)
    indices = array_ops.reshape(math_ops.range(0, sparse_size, 1), [-1, 1])
    concated = array_ops.concat([indices, sparse_labels], 1)
    dense_result = sparse_ops.sparse_to_dense(concated,
                                              [sparse_size, num_classes], 1.0,
                                              0.0)
    result = array_ops.reshape(dense_result, [height, width, num_classes])
    return result
