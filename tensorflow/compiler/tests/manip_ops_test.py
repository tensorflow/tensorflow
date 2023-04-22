# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for manip ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import googletest


class ManipOpsTest(xla_test.XLATestCase):
  """Test cases for manip ops."""

  def _testRoll(self, a, shift, axis):
    with self.session() as session:
      with self.test_scope():
        p = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        output = manip_ops.roll(a, shift, axis)
      result = session.run(output, {p: a})
      self.assertAllEqual(result, np.roll(a, shift, axis))

  def testNumericTypes(self):
    for t in self.numeric_types:
      self._testRoll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      self._testRoll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -6, 6],
          [0, 1, 2])
      self._testRoll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
          [1, 2, 3])

  def testFloatTypes(self):
    for t in self.float_types:
      self._testRoll(np.random.rand(5).astype(t), 2, 0)
      self._testRoll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
      self._testRoll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])

  def testComplexTypes(self):
    for t in self.complex_types:
      x = np.random.rand(4, 4).astype(t)
      self._testRoll(x + 1j * x, 2, 0)
      x = np.random.rand(2, 5).astype(t)
      self._testRoll(x + 1j * x, [1, 2], [1, 0])
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testRoll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])


if __name__ == "__main__":
  googletest.main()
