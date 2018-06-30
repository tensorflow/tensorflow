# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Sharded mutable dense hash table."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from tensorflow.contrib import lookup
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops


class ShardedMutableDenseHashTable(lookup.LookupInterface):
  """A sharded version of MutableDenseHashTable.

  It is designed to be interface compatible with LookupInterface and
  MutableDenseHashTable, with the exception of the export method, which is
  replaced by an export_sharded method.

  The _ShardedMutableDenseHashTable keeps `num_shards` MutableDenseHashTable
  internally. The shard is computed via the modulo operation on the key.
  """

  # TODO(andreasst): consider moving this to lookup module

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               empty_key,
               num_shards=1,
               checkpoint=True,
               name='ShardedMutableHashTable'):
    with ops.name_scope(name, 'sharded_mutable_hash_table') as scope:
      super(ShardedMutableDenseHashTable, self).__init__(key_dtype,
                                                         value_dtype, scope)
      table_shards = []
      for i in range(num_shards):
        table_shards.append(
            lookup.MutableDenseHashTable(
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                default_value=default_value,
                empty_key=empty_key,
                checkpoint=checkpoint,
                name='%s-%d-of-%d' % (name, i + 1, num_shards)))
      self._table_shards = table_shards
      # TODO(andreasst): add a value_shape() method to LookupInterface
      # pylint: disable=protected-access
      self._value_shape = self._table_shards[0]._value_shape
      # pylint: enable=protected-access

  @property
  def _num_shards(self):
    return len(self._table_shards)

  @property
  def table_shards(self):
    return self._table_shards

  def size(self, name=None):
    with ops.name_scope(name, 'sharded_mutable_hash_table_size'):
      sizes = [
          self._table_shards[i].size() for i in range(self._num_shards)
      ]
      return math_ops.add_n(sizes)

  def _shard_indices(self, keys):
    key_shape = keys.get_shape()
    if key_shape.ndims > 1:
      # If keys are a matrix (i.e. a single key is a vector), we use the first
      # element of each key vector to determine the shard.
      keys = array_ops.slice(keys, [0, 0], [key_shape[0].value, 1])
      keys = array_ops.reshape(keys, [-1])
    indices = math_ops.mod(math_ops.abs(keys), self._num_shards)
    return math_ops.cast(indices, dtypes.int32)

  def _check_keys(self, keys):
    if not keys.get_shape().is_fully_defined():
      raise ValueError('Key shape must be fully defined, got %s.' %
                       keys.get_shape())
    if keys.get_shape().ndims != 1 and keys.get_shape().ndims != 2:
      raise ValueError('Expected a vector or matrix for keys, got %s.' %
                       keys.get_shape())

  def lookup(self, keys, name=None):
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError('Signature mismatch. Keys must be dtype %s, got %s.' %
                      (self._key_dtype, keys.dtype))
    self._check_keys(keys)
    num_shards = self._num_shards
    if num_shards == 1:
      return self._table_shards[0].lookup(keys, name=name)

    shard_indices = self._shard_indices(keys)
    # TODO(andreasst): support 'keys' that are not vectors
    key_shards = data_flow_ops.dynamic_partition(keys, shard_indices,
                                                 num_shards)
    value_shards = [
        self._table_shards[i].lookup(key_shards[i], name=name)
        for i in range(num_shards)
    ]

    num_keys = keys.get_shape().dims[0]
    original_indices = math_ops.range(num_keys)
    partitioned_indices = data_flow_ops.dynamic_partition(original_indices,
                                                          shard_indices,
                                                          num_shards)
    result = data_flow_ops.dynamic_stitch(partitioned_indices, value_shards)
    result.set_shape(
        tensor_shape.TensorShape([num_keys]).concatenate(self._value_shape))
    return result

  def insert(self, keys, values, name=None):
    self._check_keys(keys)
    num_shards = self._num_shards
    if num_shards == 1:
      return self._table_shards[0].insert(keys, values, name=name)

    shard_indices = self._shard_indices(keys)
    # TODO(andreasst): support 'keys' that are not vectors
    key_shards = data_flow_ops.dynamic_partition(keys, shard_indices,
                                                 num_shards)
    value_shards = data_flow_ops.dynamic_partition(values, shard_indices,
                                                   num_shards)
    return_values = [
        self._table_shards[i].insert(key_shards[i], value_shards[i], name=name)
        for i in range(num_shards)
    ]

    return control_flow_ops.group(*return_values)

  def export_sharded(self, name=None):
    """Returns lists of the keys and values tensors in the sharded table.

    Args:
      name: name of the table.

    Returns:
      A pair of lists with the first list containing the key tensors and the
        second list containing the value tensors from each shard.
    """
    keys_list = []
    values_list = []
    for table_shard in self._table_shards:
      exported_keys, exported_values = table_shard.export(name=name)
      keys_list.append(exported_keys)
      values_list.append(exported_values)
    return keys_list, values_list
