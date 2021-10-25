# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Utility classes for testing checkpointing."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module


class CheckpointedOp(object):
  """Op with a custom checkpointing implementation.

  Defined as part of the test because the MutableHashTable Python code is
  currently in contrib.
  """

  # pylint: disable=protected-access
  def __init__(self, name, table_ref=None):
    if table_ref is None:
      self.table_ref = gen_lookup_ops.mutable_hash_table_v2(
          key_dtype=dtypes.string, value_dtype=dtypes.float32, name=name)
    else:
      self.table_ref = table_ref
    self._name = name
    if not context.executing_eagerly():
      self._saveable = CheckpointedOp.CustomSaveable(self, name)
      ops_lib.add_to_collection(ops_lib.GraphKeys.SAVEABLE_OBJECTS,
                                self._saveable)

  @property
  def name(self):
    return self._name

  @property
  def saveable(self):
    if context.executing_eagerly():
      return CheckpointedOp.CustomSaveable(self, self.name)
    else:
      return self._saveable

  def insert(self, keys, values):
    return gen_lookup_ops.lookup_table_insert_v2(self.table_ref, keys, values)

  def lookup(self, keys, default):
    return gen_lookup_ops.lookup_table_find_v2(self.table_ref, keys, default)

  def keys(self):
    return self._export()[0]

  def values(self):
    return self._export()[1]

  def _export(self):
    return gen_lookup_ops.lookup_table_export_v2(self.table_ref, dtypes.string,
                                                 dtypes.float32)

  class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):
    """A custom saveable for CheckpointedOp."""

    def __init__(self, table, name):
      tensors = table._export()
      specs = [
          saver_module.BaseSaverBuilder.SaveSpec(tensors[0], "",
                                                 name + "-keys"),
          saver_module.BaseSaverBuilder.SaveSpec(tensors[1], "",
                                                 name + "-values")
      ]
      super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

    def restore(self, restore_tensors, shapes):
      return gen_lookup_ops.lookup_table_import_v2(
          self.op.table_ref, restore_tensors[0], restore_tensors[1])
  # pylint: enable=protected-access
