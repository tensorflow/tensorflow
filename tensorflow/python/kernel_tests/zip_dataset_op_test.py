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


class ZipDatasetTest(test.TestCase):

  def testZipDataset(self):
    component_placeholders = [
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.float64)
    ]

    datasets = tuple([
        dataset_ops.Dataset.from_tensor_slices(component_placeholder)
        for component_placeholder in component_placeholders
    ])
    zipped = dataset_ops.Dataset.zip(datasets)

    iterator = zipped.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      equal_length_components = [
          np.tile(np.array([[1], [2], [3], [4]]), 20),
          np.tile(np.array([[12], [13], [14], [15]]), 22),
          np.array([37.0, 38.0, 39.0, 40.0])
      ]
      sess.run(init_op, feed_dict={ph: value for ph, value in zip(
          component_placeholders, equal_length_components)})
      for i in range(4):
        results = sess.run(get_next)
        for component, result_component in zip(
            equal_length_components, results):
          self.assertAllEqual(component[i], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      variable_length_components = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1.0, 2.0]]
      sess.run(init_op, feed_dict={ph: value for ph, value in zip(
          component_placeholders, variable_length_components)})
      for i in range(2):
        results = sess.run(get_next)
        for component, result_component in zip(
            variable_length_components, results):
          self.assertAllEqual(component[i], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testNestedZipDataset(self):
    component_placeholders = [
        array_ops.placeholder(dtypes.int64, shape=[4, 20]),
        array_ops.placeholder(dtypes.int64, shape=[4, 22]),
        array_ops.placeholder(dtypes.float64, shape=[4])
    ]

    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component_placeholder)
        for component_placeholder in component_placeholders
    ]
    zipped = dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))

    iterator = zipped.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([20], get_next[0].shape)
    self.assertEqual([22], get_next[1][0].shape)
    self.assertEqual([], get_next[1][1].shape)

    with self.test_session() as sess:
      equal_length_components = [
          np.tile(np.array([[1], [2], [3], [4]]), 20),
          np.tile(np.array([[12], [13], [14], [15]]), 22),
          np.array([37.0, 38.0, 39.0, 40.0])
      ]
      sess.run(init_op, feed_dict={ph: value for ph, value in zip(
          component_placeholders, equal_length_components)})
      for i in range(4):
        result1, (result2, result3) = sess.run(get_next)
        self.assertAllEqual(equal_length_components[0][i], result1)
        self.assertAllEqual(equal_length_components[1][i], result2)
        self.assertAllEqual(equal_length_components[2][i], result3)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
