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
"""Tests for overloaded RaggedTensor operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedElementwiseOpsTest(ragged_test_util.RaggedTensorTestCase):

  def testOrderingOperators(self):
    x = ragged_factory_ops.constant([[1, 5], [3]])
    y = ragged_factory_ops.constant([[4, 5], [1]])
    self.assertRaggedEqual((x > y), [[False, False], [True]])
    self.assertRaggedEqual((x >= y), [[False, True], [True]])
    self.assertRaggedEqual((x < y), [[True, False], [False]])
    self.assertRaggedEqual((x <= y), [[True, True], [False]])

  def testArithmeticOperators(self):
    x = ragged_factory_ops.constant([[1.0, -2.0], [8.0]])
    y = ragged_factory_ops.constant([[4.0, 4.0], [2.0]])
    self.assertRaggedEqual(abs(x), [[1.0, 2.0], [8.0]])

    self.assertRaggedEqual((-x), [[-1.0, 2.0], [-8.0]])

    self.assertRaggedEqual((x + y), [[5.0, 2.0], [10.0]])
    self.assertRaggedEqual((3.0 + y), [[7.0, 7.0], [5.0]])
    self.assertRaggedEqual((x + 3.0), [[4.0, 1.0], [11.0]])

    self.assertRaggedEqual((x - y), [[-3.0, -6.0], [6.0]])
    self.assertRaggedEqual((3.0 - y), [[-1.0, -1.0], [1.0]])
    self.assertRaggedEqual((x + 3.0), [[4.0, 1.0], [11.0]])

    self.assertRaggedEqual((x * y), [[4.0, -8.0], [16.0]])
    self.assertRaggedEqual((3.0 * y), [[12.0, 12.0], [6.0]])
    self.assertRaggedEqual((x * 3.0), [[3.0, -6.0], [24.0]])

    self.assertRaggedEqual((x / y), [[0.25, -0.5], [4.0]])
    self.assertRaggedEqual((y / x), [[4.0, -2.0], [0.25]])
    self.assertRaggedEqual((2.0 / y), [[0.5, 0.5], [1.0]])
    self.assertRaggedEqual((x / 2.0), [[0.5, -1.0], [4.0]])

    self.assertRaggedEqual((x // y), [[0.0, -1.0], [4.0]])
    self.assertRaggedEqual((y // x), [[4.0, -2.0], [0.0]])
    self.assertRaggedEqual((2.0 // y), [[0.0, 0.0], [1.0]])
    self.assertRaggedEqual((x // 2.0), [[0.0, -1.0], [4.0]])

    self.assertRaggedEqual((x % y), [[1.0, 2.0], [0.0]])
    self.assertRaggedEqual((y % x), [[0.0, -0.0], [2.0]])
    self.assertRaggedEqual((2.0 % y), [[2.0, 2.0], [0.0]])
    self.assertRaggedEqual((x % 2.0), [[1.0, 0.0], [0.0]])

  def testLogicalOperators(self):
    a = ragged_factory_ops.constant([[True, True], [False]])
    b = ragged_factory_ops.constant([[True, False], [False]])
    self.assertRaggedEqual((~a), [[False, False], [True]])

    self.assertRaggedEqual((a & b), [[True, False], [False]])
    self.assertRaggedEqual((a & True), [[True, True], [False]])
    self.assertRaggedEqual((True & b), [[True, False], [False]])

    self.assertRaggedEqual((a | b), [[True, True], [False]])
    self.assertRaggedEqual((a | False), [[True, True], [False]])
    self.assertRaggedEqual((False | b), [[True, False], [False]])

    self.assertRaggedEqual((a ^ b), [[False, True], [False]])
    self.assertRaggedEqual((a ^ True), [[False, False], [True]])
    self.assertRaggedEqual((True ^ b), [[False, True], [True]])

  def testDummyOperators(self):
    a = ragged_factory_ops.constant([[True, True], [False]])
    with self.assertRaisesRegexp(TypeError,
                                 'RaggedTensor may not be used as a boolean.'):
      bool(a)
    with self.assertRaisesRegexp(TypeError,
                                 'RaggedTensor may not be used as a boolean.'):
      if a:
        pass


if __name__ == '__main__':
  googletest.main()
