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
"""Tests for the RangeDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RangeDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _iterator_checkpoint_prefix_local(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def _save_op(self, iterator_resource):
    iterator_state_variant = gen_dataset_ops.serialize_iterator(
        iterator_resource)
    save_op = io_ops.write_file(
        self._iterator_checkpoint_prefix_local(),
        parsing_ops.serialize_tensor(iterator_state_variant))
    return save_op

  def _restore_op(self, iterator_resource):
    iterator_state_variant = parsing_ops.parse_tensor(
        io_ops.read_file(self._iterator_checkpoint_prefix_local()),
        dtypes.variant)
    restore_op = gen_dataset_ops.deserialize_iterator(iterator_resource,
                                                      iterator_state_variant)
    return restore_op

  def testSaveRestore(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, self.evaluate(get_next))
        self.evaluate(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        self.evaluate(init_op)
        self.evaluate(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, self.evaluate(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, self.evaluate(get_next))
        self.evaluate(save_op)
        self.evaluate(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, self.evaluate(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          self.evaluate(get_next)

  def _build_range_dataset(self, start, stop):
    return dataset_ops.Dataset.range(start, stop)

  def testRangeCore(self):
    start = 2
    stop = 10
    stop_1 = 8
    self.run_core_tests(lambda: self._build_range_dataset(start, stop),
                        lambda: self._build_range_dataset(start, stop_1),
                        stop - start)


if __name__ == "__main__":
  test.main()
