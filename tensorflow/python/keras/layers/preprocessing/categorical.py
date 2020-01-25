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
"""Keras categorical preprocessing layers."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import lookup_ops


class CategoryLookup(Layer):
  """Category lookup layer.

  This layer looks up tokens (int or string) in a vocabulary table,
  and return their indices (int). It converts a sequence of int or string to a
  sequence of int.

  Attributes:
    vocabulary: The vocabulary to lookup the input. If it is a file, it
      represents the source vocab file; If it is a list/tuple, it represents the
      source vocab list. If it is None, the vocabulary can later be set.
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. This is used when `adapt`
      is called.
    num_oov_tokens: Non-negative integer. The number of out-of-vocab tokens. All
      out-of-vocab inputs will be assigned IDs in the range of [0,
      num_oov_tokens) based on a hash.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.
  Input shape: A string or int tensor of shape `[batch_size, d1, ..., dm]`
  Output shape: An int tensor of shape `[batch_size, d1, .., dm]`
  Example: Consider a batch of a single input sample, `[["a", "c", "d", "a",
    "x"]]`. Let's say the vocabulary is `["a", "b", "c", "d"]` and a single OOV
    token is used (`num_oov_tokens=1`). Then the corresponding output is `[[1,
    3, 4, 1, 0]]`. 0 stands for an OOV token.
  """

  def __init__(self,
               max_tokens=None,
               num_oov_tokens=1,
               vocabulary=None,
               name=None,
               **kwargs):
    if max_tokens is not None:
      raise ValueError('`max_tokens` and `adapt` is not supported yet.')
    if vocabulary is None:
      raise ValueError('for now, you must pass a `vocabulary` argument')
    self.max_tokens = max_tokens
    self.num_oov_tokens = num_oov_tokens
    self.vocabulary = vocabulary
    super(CategoryLookup, self).__init__(name, **kwargs)

  def __call__(self, inputs, *args, **kwargs):
    if isinstance(inputs, (np.ndarray, float, int)):
      inputs = ops.convert_to_tensor(inputs)
    self._input_dtype = inputs.dtype
    return super(CategoryLookup, self).__call__(inputs, *args, **kwargs)

  def build(self, input_shape):
    # categorical with vocabulary list.
    if isinstance(self.vocabulary, (tuple, list, np.ndarray)):
      self.table = lookup_ops.index_table_from_tensor(
          vocabulary_list=self.vocabulary,
          num_oov_buckets=self.num_oov_tokens,
          dtype=self._input_dtype)
    # categorical with vocabulary file.
    elif self.vocabulary:
      self.table = lookup_ops.index_table_from_file(
          vocabulary_file=self.vocabulary,
          num_oov_buckets=self.num_oov_tokens,
          key_dtype=self._input_dtype)

  def call(self, inputs):
    return self.table.lookup(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64
    if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    else:
      return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)
