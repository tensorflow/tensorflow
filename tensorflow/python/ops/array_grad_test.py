# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Python ops defined in array_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class TileGradientTest(test.TestCase):

  def testTileGradientWithDenseGrad(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    inputs = array_ops.reshape(inputs, [-1, 1, 1])
    outputs = array_ops.tile(inputs, [2, 3, 4])
    with self.test_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  def testTileGradientWithSparseGrad(self):
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    inputs = array_ops.reshape(inputs, [-1, 1, 1])
    outputs = array_ops.gather(array_ops.tile(inputs, [3, 4, 2]),
                               [1, 5, 9, 3, 7, 2, 2, 2])
    with self.test_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()
