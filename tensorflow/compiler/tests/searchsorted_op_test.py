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
"""Test for XLA implementation of tf.searchsorted."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SearchSorteddOpTest(xla_test.XLATestCase):

  def test1D(self):
    # Test against NumPy implementation (which is 1D only).
    np.random.seed(1)
    for side in ['left', 'right']:
      for dtype in [np.float32, np.int32]:
        values = np.random.uniform(
            low=-1000, high=1000, size=(10,)).astype(dtype)
        unsorted = np.random.uniform(
            low=-1000, high=1000, size=(20,)).astype(dtype)

        sorted_sequence = np.sort(unsorted)
        np_ans = np.searchsorted(sorted_sequence, values, side=side)

        with self.session() as session:
          with self.test_scope():
            tf_ans = array_ops.searchsorted(sorted_sequence, values, side=side)
          tf_out = session.run(tf_ans)
          self.assertAllEqual(np_ans, tf_out)

  def testNanValue(self):
    for dtype in self.float_types:
      sorted_sequence = np.array([[1, 3, 5, 7, 9]], dtype)
      values = np.array([[np.nan]], dtype)

      self._test2DExample(
          dtype,
          'left',
          sorted_sequence,
          values,
          np.array([[0]], dtype=np.int32),
      )

      self._test2DExample(
          dtype,
          'right',
          sorted_sequence,
          values,
          np.array([[5]], dtype=np.int32),
      )

  def _test2DExample(self, dtype, side, sorted_sequence, values, correct_ans):

    with self.session() as session:
      with self.test_scope():
        tf_ans = array_ops.searchsorted(sorted_sequence, values, side=side)
      tf_out = session.run(tf_ans)
      self.assertAllEqual(correct_ans, tf_out)

  def testLowerBound2DExample(self):
    # 2D TensorFlow documentation example.
    for dtype in self.float_types | self.int_types:
      sorted_sequence = np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], dtype)
      values = np.array([[2, 4, 9], [0, 2, 6]], dtype)
      correct_ans = np.array([[1, 2, 2], [0, 1, 5]], dtype)
      self._test2DExample(dtype, 'left', sorted_sequence, values, correct_ans)

  def testUpperBound2DExample(self):
    # 2D TensorFlow documentation example.
    for dtype in self.float_types | self.int_types:
      sorted_sequence = np.array([[0, 3, 9, 9, 10], [1, 2, 3, 4, 5]], dtype)
      values = np.array([[2, 4, 9], [0, 2, 6]], dtype)
      correct_ans = np.array([[1, 2, 4], [0, 2, 5]], dtype)
      self._test2DExample(dtype, 'right', sorted_sequence, values, correct_ans)


  def testLowerBoundNaN(self):
    sorted_sequence = np.array(
        [[2.0, 4.0, 8.0, 16.0, 32.0, 64.0]],
        dtype=np.float32)
    values = np.array(
        [[np.nan, 8.0]],
        dtype=np.float32)
    expected = np.searchsorted(
        sorted_sequence[0],
        values[0],
        side="left")
    with self.session() as session:
      with self.test_scope():
        tf_ans = array_ops.searchsorted(
            sorted_sequence,
            values,
            side="left")
      tf_out = session.run(tf_ans)
    self.assertAllEqual(expected.reshape(1, -1), tf_out)

  def testUpperBoundNaN(self):
    sorted_sequence = np.array(
        [[2.0, 4.0, 8.0, 16.0, 32.0, 64.0]],
        dtype=np.float32)

    values = np.array(
        [[np.nan, 8.0]],
        dtype=np.float32)

    expected = np.searchsorted(
        sorted_sequence[0],
        values[0],
        side="right")

    with self.session() as session:
      with self.test_scope():
        tf_ans = array_ops.searchsorted(
            sorted_sequence,
            values,
            side="right")
      tf_out = session.run(tf_ans)

    self.assertAllEqual(expected.reshape(1, -1), tf_out)

if __name__ == '__main__':
  test.main()
