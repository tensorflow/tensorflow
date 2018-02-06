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
"""Test RangeDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import counter
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.data.python.ops import enumerate_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RangeDatasetTest(test.TestCase):

  def testStop(self):
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={stop: 5})
      for i in range(5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStop(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 5})
      for i in range(2, 5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStopStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: 2})
      for i in range(2, 10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testZeroStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer

    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={start: 2, stop: 10, step: 0})

  def testNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: -1})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(2, 10, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStart(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithPositiveStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: -1})
      for i in range(10, 2, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEnumerateDataset(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    iterator = (dataset_ops.Dataset.from_tensor_slices(components).apply(
        enumerate_ops.enumerate_dataset(start)).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int64, get_next[0].dtype)
    self.assertEqual((), get_next[0].shape)
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [t.shape for t in get_next[1]])

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual((20, (b"a", 1, 37.0)), sess.run(get_next))
      self.assertEqual((21, (b"b", 2, 38.0)), sess.run(get_next))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCounter(self):
    """Test dataset construction using `count`."""
    iterator = (counter.Counter(start=3, step=4)
                .make_one_shot_iterator())
    get_next = iterator.get_next()
    self.assertEqual([], get_next.shape.as_list())
    self.assertEqual(dtypes.int64, get_next.dtype)

    negative_iterator = (counter.Counter(start=0, step=-1)
                         .make_one_shot_iterator())
    negative_get_next = negative_iterator.get_next()

    with self.test_session() as sess:
      self.assertEqual(3, sess.run(get_next))
      self.assertEqual(3 + 4, sess.run(get_next))
      self.assertEqual(3 + 2 * 4, sess.run(get_next))

      self.assertEqual(0, sess.run(negative_get_next))
      self.assertEqual(-1, sess.run(negative_get_next))
      self.assertEqual(-2, sess.run(negative_get_next))


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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

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
