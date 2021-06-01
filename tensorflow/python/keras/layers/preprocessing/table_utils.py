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

import collections
import os
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile


class TableHandler(object):
  """Wrapper object that holds a lookup table and provides accessors."""

  def __init__(self,
               table,
               oov_tokens=None,
               mask_token=None,
               mask_value=0):
    self.table = table
    self.mutable = isinstance(table, lookup_ops.MutableHashTable)
    self.mask_token = mask_token
    self.mask_value = mask_value

    if oov_tokens is None:
      self.oov_tokens = oov_tokens
    else:
      if not isinstance(oov_tokens, (list, tuple, np.ndarray)):
        oov_tokens = [oov_tokens]
      self.oov_tokens = math_ops.cast(oov_tokens, table._value_dtype)  # pylint: disable=protected-access

  def table_size(self):
    return self.table.size().numpy()

  def clear(self):
    if not self.mutable:
      return RuntimeError("Unable to clear a statically-backed table.")

    keys, _ = self.table.export()
    self.table.remove(keys)

  def insert(self, keys, values):
    """Insert values into the backed table."""
    if not self.mutable:
      raise RuntimeError("Unable to insert into a statically-backed table.")

    if len(values) != len(keys):
      raise RuntimeError("Size mismatch between values and key arrays. "
                         "Keys had size %s, values had size %s." %
                         (len(keys), len(values)))
    keys = ops.convert_to_tensor_v2_with_dispatch(
        keys, dtype=self.table._key_dtype)  # pylint: disable=protected-access
    values = ops.convert_to_tensor_v2_with_dispatch(
        values, dtype=self.table._value_dtype)  # pylint: disable=protected-access
    if values.shape.ndims != 1:
      raise ValueError("`values` must be 1-dimensional, got an input with "
                       " %s dimensions." % values.shape.ndims)
    self.table.insert(keys, values)

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

  def _lookup_and_mask(self, inputs):
    """Return a lookup with any location with the mask_token masked to 0."""
    lookups = self.table.lookup(inputs)
    # If we don't need to handle masking, return the lookup values directly.
    if self.mask_token is None:
      return lookups

    # Inject 0s wherever the mask token was in the inputs.
    mask_locations = math_ops.equal(inputs, self.mask_token)
    return array_ops.where_v2(
        mask_locations,
        math_ops.cast(self.mask_value, self.table._value_dtype),  # pylint: disable=protected-access
        lookups)  # pylint: disable=protected-access

  def _ragged_lookup(self, inputs):
    """Perform a table lookup on a ragged tensor."""
    # The table lookup ops don't natively support ragged tensors, so if we have
    # a RT we need to use map_flat_values to look up every element.
    indexed_data = ragged_functional_ops.map_flat_values(
        self._lookup_and_mask, inputs)
    indexed_data = ragged_functional_ops.map_flat_values(
        self._replace_oov_buckets, inputs, indexed_data)
    # table.lookup is not shape-preserving, so we need to set the shape here.
    indexed_data._set_shape(inputs.shape)  # pylint: disable=protected-access
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return array_ops.identity(indexed_data)

  def _sparse_lookup(self, inputs):
    """Perform a table lookup on a sparse tensor."""
    values = self._lookup_and_mask(inputs.values)
    values = self._replace_oov_buckets(inputs.values, values)
    indexed_data = sparse_tensor.SparseTensor(inputs.indices, values,
                                              inputs.dense_shape)
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return array_ops.identity(indexed_data)

  def _tensor_lookup(self, inputs):
    """Perform a table lookup on a tf.tensor."""
    values = self._lookup_and_mask(inputs)
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

    if tf_utils.is_ragged(inputs):
      if isinstance(inputs, ragged_tensor_value.RaggedTensorValue):
        flat_values = ops.convert_to_tensor_v2_with_dispatch(
            value=inputs.flat_values, name="flat_values")
        inputs = ragged_tensor.RaggedTensor.from_nested_row_splits(
            flat_values, inputs.nested_row_splits, validate=False)
      return self._ragged_lookup(inputs)

    # For normal tensor inputs
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    return self._tensor_lookup(inputs)


def num_tokens_in_file(vocabulary_path):
  """Count the number of lines in a vocab file to get the number of tokens."""
  num_tokens = 0
  with gfile.GFile(vocabulary_path, "r") as reader:
    text = reader.readline()
    while text:
      num_tokens += 1
      text = reader.readline()

  return num_tokens


def get_vocabulary_from_file(vocabulary_path, encoding="utf-8"):
  """Read a vocabulary in from a file."""
  vocab = []
  with gfile.GFile(vocabulary_path, "r") as reader:
    while True:
      # Get the next line (incl. \n), and break if nothing is left to read.
      text = reader.readline()
      if not text:
        break

      # Convert the raw text and strip whitespace.
      if isinstance(text, str):
        token = text
      elif isinstance(text, bytes):
        token = text.decode(encoding, "ignore")
      token = token.rstrip(os.linesep)
      vocab.append(token)
  return vocab


def find_repeated_tokens(vocabulary):
  """Return all repeated tokens in a vocabulary."""
  vocabulary_set = set(vocabulary)
  if len(vocabulary) != len(vocabulary_set):
    return [
        item for item, count in collections.Counter(vocabulary).items()
        if count > 1
    ]
  else:
    return []
