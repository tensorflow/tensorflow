# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for working with tf.lookup tables in Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile


class TableHandler(object):
  """Wrapper object that holds a lookup table and provides accessors."""

  def __init__(self, table, oov_tokens=None, use_v1_apis=False):
    self.table = table
    self.use_v1_apis = use_v1_apis
    if oov_tokens is None:
      self.oov_tokens = oov_tokens
    else:
      if not isinstance(oov_tokens, (list, tuple, np.ndarray)):
        oov_tokens = [oov_tokens]
      self.oov_tokens = math_ops.cast(oov_tokens, table._value_dtype)  # pylint: disable=protected-access

  def data(self):
    keys, values = self.table.export()
    return (self._eval(keys), self._eval(values))

  def vocab_size(self):
    return self._eval(self.table.size())

  def clear(self):
    keys, _ = self.table.export()
    self._run(self.table.remove(keys))

  def insert(self, keys, values):
    if len(values) != len(keys):
      raise RuntimeError("Size mismatch between values and key arrays. "
                         "Keys had size %s, values had size %s." %
                         (len(keys), len(values)))
    self._run(self.table.insert(keys, values))

  def _replace_oov_buckets(self, inputs, lookups):
    """Replace the default OOV value with one of the OOV bucket values."""
    if self.oov_tokens is None:
      return lookups

    num_oov_elements = self.oov_tokens.shape.num_elements()
    if inputs.dtype.is_integer:
      oov_indices = math_ops.floormod(inputs, num_oov_elements)
    else:
      oov_indices = string_ops.string_to_hash_bucket_fast(
          inputs, num_buckets=num_oov_elements)

    oov_values = array_ops.gather(self.oov_tokens, oov_indices)
    oov_locations = math_ops.equal(lookups, self.table._default_value)  # pylint: disable=protected-access

    return array_ops.where(oov_locations, oov_values, lookups)

  def _ragged_lookup(self, inputs):
    """Perform a table lookup on a ragged tensor."""
    # The table lookup ops don't natively support ragged tensors, so if we have
    # a RT we need to use map_flat_values to look up every element.
    indexed_data = ragged_functional_ops.map_flat_values(
        self.table.lookup, inputs)
    indexed_data = ragged_functional_ops.map_flat_values(
        self._replace_oov_buckets, inputs, indexed_data)
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return array_ops.identity(indexed_data)

  def _sparse_lookup(self, inputs):
    """Perform a table lookup on a sparse tensor."""
    values = self.table.lookup(inputs.values)
    values = self._replace_oov_buckets(inputs.values, values)
    indexed_data = sparse_tensor.SparseTensor(inputs.indices, values,
                                              inputs.dense_shape)
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return array_ops.identity(indexed_data)

  def _tensor_lookup(self, inputs):
    """Perform a table lookup on a tf.tensor."""
    values = self.table.lookup(inputs)
    indexed_data = self._replace_oov_buckets(inputs, values)
    # (b/149446477): output does not preserve input shape.
    indexed_data.set_shape(inputs.shape)
    return indexed_data

  def lookup(self, inputs):
    """Perform a table lookup."""
    # Sparse tensors don't play nicely with tensor conversion, so we handle
    # them before attempting to convert lists or arrays to tensors.
    if isinstance(
        inputs, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      return self._sparse_lookup(inputs)

    # Try to convert lists/arrays to tensors or RaggedTensors.
    inputs = ragged_tensor.convert_to_tensor_or_ragged_tensor(inputs)

    # Run the lookup operation on the converted tensor.
    if ragged_tensor.is_ragged(inputs):
      return self._ragged_lookup(inputs)
    else:
      return self._tensor_lookup(inputs)

  def _eval(self, tensor):
    if self.use_v1_apis:
      return K.get_session().run(tensor)
    else:
      return tensor.numpy()

  def _run(self, op):
    if self.use_v1_apis:
      K.get_session().run(op)


def get_vocabulary_from_file(vocabulary_path, encoding="utf-8"):
  """Read a vocabulary in from a file."""
  vocab = []
  with gfile.GFile(vocabulary_path, "r") as reader:
    while True:
      # Get the next line, and break if it is None.
      text = reader.readline()
      if not text:
        break

      # Convert the raw text and strip whitespace.
      if isinstance(text, str):
        token = text
      elif isinstance(text, bytes):
        token = text.decode(encoding, "ignore")
      token = token.strip()
      vocab.append(token)
  return vocab


def validate_vocabulary_is_unique(vocabulary):
  """Validate that a vocabulary contains no repeated tokens."""
  vocabulary_set = set(vocabulary)
  if len(vocabulary) != len(vocabulary_set):
    repeated_items = [
        item for item, count in collections.Counter(vocabulary).items()
        if count > 1
    ]
    raise ValueError("The passed vocabulary has at least one repeated "
                     "term. Please uniquify your dataset. The repeated terms "
                     "are %s" % repeated_items)


def assert_same_type(expected_type, values, value_name):
  """Assert that 'values' is of type 'expected_type'."""
  if dtypes.as_dtype(expected_type) != dtypes.as_dtype(values.dtype):
    raise RuntimeError("Expected %s type %s, got %s" %
                       (value_name, expected_type, values.dtype))


def convert_to_ndarray(x, dtype=None):
  """Convert 'x' to a numpy array."""
  array = np.array(x) if isinstance(x, (list, tuple)) else x
  if dtype not in (None, dtypes.string):
    # If the dtype is an integer, we do permissive casting. This allows
    # users to examine int32 data if the dtype is int64 without trouble.
    np_dtype = dtypes.as_dtype(dtype).as_numpy_dtype
    if np.can_cast(array.dtype, np_dtype):
      array = array.astype(np_dtype, casting="safe")
  return array

