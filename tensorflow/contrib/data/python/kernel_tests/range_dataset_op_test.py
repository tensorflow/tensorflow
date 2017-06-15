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

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
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

    iterator = (dataset_ops.Dataset.from_tensor_slices(components).enumerate(
        start=start).make_initializable_iterator())
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


if __name__ == "__main__":
  test.main()
