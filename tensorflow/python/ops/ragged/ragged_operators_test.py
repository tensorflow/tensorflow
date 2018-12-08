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
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


@test_util.run_v1_only('b/120545219')
class RaggedElementwiseOpsTest(test_util.TensorFlowTestCase):
  # @TODO(edloper): Test right-handed versions of operators once we add
  # broadcasting support for elementwise ops.

  def testOrderingOperators(self):
    x = ragged.constant([[1, 5], [3]])
    y = ragged.constant([[4, 5], [1]])
    with self.test_session():
      self.assertEqual((x > y).eval().tolist(), [[False, False], [True]])
      self.assertEqual((x >= y).eval().tolist(), [[False, True], [True]])
      self.assertEqual((x < y).eval().tolist(), [[True, False], [False]])
      self.assertEqual((x <= y).eval().tolist(), [[True, True], [False]])

  def assertEqual(self, a, b):
    if a != b:
      print('%30s %s' % (b, a))

  def testArithmeticOperators(self):
    x = ragged.constant([[1.0, -2.0], [8.0]])
    y = ragged.constant([[4.0, 4.0], [2.0]])
    with self.test_session():
      self.assertEqual(abs(x).eval().tolist(), [[1.0, 2.0], [8.0]])

      self.assertEqual((-x).eval().tolist(), [[-1.0, 2.0], [-8.0]])

      self.assertEqual((x + y).eval().tolist(), [[5.0, 2.0], [10.0]])
      self.assertEqual((3.0 + y).eval().tolist(), [[7.0, 7.0], [5.0]])
      self.assertEqual((x + 3.0).eval().tolist(), [[4.0, 1.0], [11.0]])

      self.assertEqual((x - y).eval().tolist(), [[-3.0, -6.0], [6.0]])
      self.assertEqual((3.0 - y).eval().tolist(), [[-1.0, -1.0], [1.0]])
      self.assertEqual((x + 3.0).eval().tolist(), [[4.0, 1.0], [11.0]])

      self.assertEqual((x * y).eval().tolist(), [[4.0, -8.0], [16.0]])
      self.assertEqual((3.0 * y).eval().tolist(), [[12.0, 12.0], [6.0]])
      self.assertEqual((x * 3.0).eval().tolist(), [[3.0, -6.0], [24.0]])

      self.assertEqual((x / y).eval().tolist(), [[0.25, -0.5], [4.0]])
      self.assertEqual((y / x).eval().tolist(), [[4.0, -2.0], [0.25]])
      self.assertEqual((2.0 / y).eval().tolist(), [[0.5, 0.5], [1.0]])
      self.assertEqual((x / 2.0).eval().tolist(), [[0.5, -1.0], [4.0]])

      self.assertEqual((x // y).eval().tolist(), [[0.0, -1.0], [4.0]])
      self.assertEqual((y // x).eval().tolist(), [[4.0, -2.0], [0.0]])
      self.assertEqual((2.0 // y).eval().tolist(), [[0.0, 0.0], [1.0]])
      self.assertEqual((x // 2.0).eval().tolist(), [[0.0, -1.0], [4.0]])

      self.assertEqual((x % y).eval().tolist(), [[1.0, 2.0], [0.0]])
      self.assertEqual((y % x).eval().tolist(), [[0.0, -0.0], [2.0]])
      self.assertEqual((2.0 % y).eval().tolist(), [[2.0, 2.0], [0.0]])
      self.assertEqual((x % 2.0).eval().tolist(), [[1.0, 0.0], [0.0]])

  def testLogicalOperators(self):
    a = ragged.constant([[True, True], [False]])
    b = ragged.constant([[True, False], [False]])
    with self.test_session():
      self.assertEqual((~a).eval().tolist(), [[False, False], [True]])

      self.assertEqual((a & b).eval().tolist(), [[True, False], [False]])
      self.assertEqual((a & True).eval().tolist(), [[True, True], [False]])
      self.assertEqual((True & b).eval().tolist(), [[True, False], [False]])

      self.assertEqual((a | b).eval().tolist(), [[True, True], [False]])
      self.assertEqual((a | False).eval().tolist(), [[True, True], [False]])
      self.assertEqual((False | b).eval().tolist(), [[True, False], [False]])

      self.assertEqual((a ^ b).eval().tolist(), [[False, True], [False]])
      self.assertEqual((a ^ True).eval().tolist(), [[False, False], [True]])
      self.assertEqual((True ^ b).eval().tolist(), [[False, True], [True]])

  def testDummyOperators(self):
    a = ragged.constant([[True, True], [False]])
    with self.assertRaisesRegexp(TypeError,
                                 'RaggedTensor may not be used as a boolean.'):
      bool(a)
    with self.assertRaisesRegexp(TypeError,
                                 'RaggedTensor may not be used as a boolean.'):
      if a:
        pass


if __name__ == '__main__':
  googletest.main()
