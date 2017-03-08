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
# ==============================================================================

"""Tests for tensorflow.ops.reverse_sequence_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class WhereOpTest(tf.test.TestCase):

  def _testWhere(self, x, truth, expected_err_re=None):
    with self.test_session():
      ans = tf.where(x)
      self.assertEqual([None, x.ndim], ans.get_shape().as_list())
      if expected_err_re is None:
        tf_ans = ans.eval()
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval()

  def testWrongNumbers(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.where([False, True], [1, 2], None)
      with self.assertRaises(ValueError):
        tf.where([False, True], None, [1, 2])

  def testBasicMat(self):
    x = np.asarray([[True, False], [True, False]])

    # Ensure RowMajor mode
    truth = np.asarray([[0, 0], [1, 0]], dtype=np.int64)

    self._testWhere(x, truth)

  def testBasic3Tensor(self):
    x = np.asarray(
        [[[True, False], [True, False]], [[False, True], [False, True]],
         [[False, False], [False, True]]])

    # Ensure RowMajor mode
    truth = np.asarray(
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [2, 1, 1]],
        dtype=np.int64)

    self._testWhere(x, truth)

  def testThreeArgument(self):
    x = np.array([[-2, 3, -1], [1, -3, -3]])
    np_val = np.where(x > 0, x*x, -x)
    with self.test_session():
      tf_val = tf.where(tf.constant(x) > 0, x*x, -x).eval()
    self.assertAllEqual(tf_val, np_val)

if __name__ == "__main__":
  tf.test.main()
