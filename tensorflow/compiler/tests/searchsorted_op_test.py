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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


if __name__ == '__main__':
  test.main()
