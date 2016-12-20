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

"""Tests for learn.dataframe.transforms.boolean_mask."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks


class BooleanMaskTestCase(tf.test.TestCase):
  """Test class for `BooleanMask`."""

  def testDense(self):
    dense_shape = [10, 5]

    np.random.seed(1234)
    # Create a random Tensor with dense_shape.
    random_array = np.random.randn(*dense_shape)
    # Randomly choose ~1/2 of the rows.
    mask = np.random.randn(dense_shape[0]) > 0.5
    expected_result = random_array[mask]

    dense_series = mocks.MockSeries("dense_series", tf.constant(random_array))
    mask_series = mocks.MockSeries("mask", tf.constant(mask))
    masked = dense_series.select_rows(mask_series)

    with self.test_session() as sess:
      actual_result = sess.run(masked.build())

    np.testing.assert_almost_equal(expected_result, actual_result)

  def testSparse(self):
    indices = [[0, 0, 0], [0, 1, 1], [5, 1, 2], [3, 0, 2], [2, 4, 1], [3, 5, 1],
               [7, 3, 2]]
    values = list(range(len(indices)))
    shape = [max(x) + 1 for x in zip(*indices)]

    np.random.seed(1234)

    # Randomly choose ~1/2 of the rows.
    mask = np.random.randn(shape[0]) > 0.5

    index_value_pairs = [[ind, val] for ind, val in zip(indices, values)
                         if mask[ind[0]]]
    expected_indices, expected_values = zip(*index_value_pairs)

    sparse_series = mocks.MockSeries("sparse_series",
                                     tf.SparseTensor(indices, values, shape))
    mask_series = mocks.MockSeries("mask", tf.constant(mask))
    masked = sparse_series.select_rows(mask_series)

    with self.test_session() as sess:
      actual = sess.run(masked.build())

    np.testing.assert_array_equal(expected_indices, actual.indices)
    np.testing.assert_array_equal(expected_values, actual.values)
    np.testing.assert_array_equal(shape, actual.dense_shape)

if __name__ == "__main__":
  tf.test.main()
