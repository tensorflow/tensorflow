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
"""Tests for tf.contrib.kfac.loss_functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import loss_functions
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class InsertSliceInZerosTest(test.TestCase):

  def testBadShape(self):
    bad_shaped_ones = array_ops.ones(shape=[1, 3])  # n.b. shape[1] != 1
    with self.assertRaises(ValueError):
      loss_functions.insert_slice_in_zeros(bad_shaped_ones, 1, 42, 17)

  def test3d(self):
    input_tensor = constant_op.constant([[[1, 2]], [[3, 4]]])
    expected_output_array = [[[1, 2], [0, 0]], [[3, 4], [0, 0]]]
    op = loss_functions.insert_slice_in_zeros(input_tensor, 1, 2, 0)
    with self.test_session() as sess:
      actual_output_array = sess.run(op)
    self.assertAllEqual(expected_output_array, actual_output_array)


if __name__ == "__main__":
  test.main()
