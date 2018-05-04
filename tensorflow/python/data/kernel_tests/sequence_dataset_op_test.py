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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SequenceDatasetTest(test.TestCase):

  def testRepeatTensorDataset(self):
    """Test a dataset that repeats its input multiple times."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    # This placeholder can be fed when dataset-definition subgraph
    # runs (i.e. `init_op` below) to configure the number of
    # repetitions used in a particular iterator.
    count_placeholder = array_ops.placeholder(dtypes.int64, shape=[])

    iterator = (dataset_ops.Dataset.from_tensors(components)
                .repeat(count_placeholder).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      # Test a finite repetition.
      sess.run(init_op, feed_dict={count_placeholder: 3})
      for _ in range(3):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test a different finite repetition.
      sess.run(init_op, feed_dict={count_placeholder: 7})
      for _ in range(7):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test an empty repetition.
      sess.run(init_op, feed_dict={count_placeholder: 0})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test an infinite repetition.
      # NOTE(mrry): There's not a good way to test that the sequence
      # actually is infinite.
      sess.run(init_op, feed_dict={count_placeholder: -1})
      for _ in range(17):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)

  def testTakeTensorDataset(self):
    components = (np.arange(10),)
    count_placeholder = array_ops.placeholder(dtypes.int64, shape=[])

    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .take(count_placeholder).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      # Take fewer than input size
      sess.run(init_op, feed_dict={count_placeholder: 4})
      for i in range(4):
        results = sess.run(get_next)
        self.assertAllEqual(results, components[0][i:i+1])

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Take more than input size
      sess.run(init_op, feed_dict={count_placeholder: 25})
      for i in range(10):
        results = sess.run(get_next)
        self.assertAllEqual(results, components[0][i:i+1])

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Take all of input
      sess.run(init_op, feed_dict={count_placeholder: -1})
      for i in range(10):
        results = sess.run(get_next)
        self.assertAllEqual(results, components[0][i:i+1])

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Take nothing
      sess.run(init_op, feed_dict={count_placeholder: 0})

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSkipTensorDataset(self):
    components = (np.arange(10),)
    count_placeholder = array_ops.placeholder(dtypes.int64, shape=[])

    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .skip(count_placeholder).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      # Skip fewer than input size, we should skip
      # the first 4 elements and then read the rest.
      sess.run(init_op, feed_dict={count_placeholder: 4})
      for i in range(4, 10):
        results = sess.run(get_next)
        self.assertAllEqual(results, components[0][i:i+1])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Skip more than input size: get nothing.
      sess.run(init_op, feed_dict={count_placeholder: 25})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Skip exactly input size.
      sess.run(init_op, feed_dict={count_placeholder: 10})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Set -1 for 'count': skip the entire dataset.
      sess.run(init_op, feed_dict={count_placeholder: -1})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Skip nothing
      sess.run(init_op, feed_dict={count_placeholder: 0})
      for i in range(0, 10):
        results = sess.run(get_next)
        self.assertAllEqual(results, components[0][i:i+1])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testRepeatRepeatTensorDataset(self):
    """Test the composition of repeat datasets."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    inner_count = array_ops.placeholder(dtypes.int64, shape=[])
    outer_count = array_ops.placeholder(dtypes.int64, shape=[])

    iterator = (dataset_ops.Dataset.from_tensors(components).repeat(inner_count)
                .repeat(outer_count).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={inner_count: 7, outer_count: 14})
      for _ in range(7 * 14):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testRepeatEmptyDataset(self):
    """Test that repeating an empty dataset does not hang."""
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(10).skip(10)
                .repeat(-1).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
