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
"""Test cases for operators with no arguments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class NullaryOpsTest(xla_test.XLATestCase):

  def _testNullary(self, op, expected):
    with self.session() as session:
      with self.test_scope():
        output = op()
      result = session.run(output)
      self.assertAllClose(result, expected, rtol=1e-3)

  def testNoOp(self):
    with self.session():
      with self.test_scope():
        output = control_flow_ops.no_op()
      # This should not crash.
      output.run()

  def testConstants(self):
    for dtype in self.numeric_types:
      constants = [
          dtype(42),
          np.array([], dtype=dtype),
          np.array([1, 2], dtype=dtype),
          np.array([7, 7, 7, 7, 7], dtype=dtype),
          np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
          np.array([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]],
                   dtype=dtype),
          np.array([[[]], [[]]], dtype=dtype),
          np.array([[[[1]]]], dtype=dtype),
      ]
      for c in constants:
        self._testNullary(lambda c=c: constant_op.constant(c), expected=c)

  def testComplexConstants(self):
    for dtype in self.complex_types:
      constants = [
          dtype(42 + 3j),
          np.array([], dtype=dtype),
          np.ones([50], dtype=dtype) * (3 + 4j),
          np.array([1j, 2 + 1j], dtype=dtype),
          np.array([[1, 2j, 7j], [4, 5, 6]], dtype=dtype),
          np.array([[[1, 2], [3, 4 + 6j], [5, 6]],
                    [[10 + 7j, 20], [30, 40], [50, 60]]],
                   dtype=dtype),
          np.array([[[]], [[]]], dtype=dtype),
          np.array([[[[1 + 3j]]]], dtype=dtype),
      ]
      for c in constants:
        self._testNullary(lambda c=c: constant_op.constant(c), expected=c)


if __name__ == "__main__":
  googletest.main()
