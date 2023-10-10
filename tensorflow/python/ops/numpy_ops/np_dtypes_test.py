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
"""Tests for tf-numpy dtype utilities."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import test


class DTypeTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([False, True])
  def testAllowF64False(self, prefer_f32):
    np_dtypes.set_allow_float64(False)
    np_dtypes.set_prefer_float32(prefer_f32)
    self.assertEqual(dtypes.float32, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float32,
                     np_dtypes._result_type(np.zeros([], np.float64), 1.1))

  def testAllowF64TruePreferF32False(self):
    np_dtypes.set_allow_float64(True)
    np_dtypes.set_prefer_float32(False)
    self.assertEqual(dtypes.float64, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float64, np_dtypes._result_type(1.1))
    self.assertEqual(dtypes.complex128, np_dtypes._result_type(1.j))

  def testAllowF64TruePreferF32True(self):
    np_dtypes.set_allow_float64(True)
    np_dtypes.set_prefer_float32(True)
    self.assertEqual(dtypes.float32, np_dtypes.default_float_type())
    self.assertEqual(dtypes.float32, np_dtypes._result_type(1.1))
    self.assertEqual(dtypes.float64,
                     np_dtypes._result_type(np.zeros([], np.float64), 1.1))
    self.assertEqual(dtypes.complex64, np_dtypes._result_type(1.1j))
    self.assertEqual(dtypes.complex128,
                     np_dtypes._result_type(np.zeros([], np.complex128), 1.1j))
    self.assertEqual(dtypes.complex64,
                     np_dtypes._result_type(np.zeros([], np.float32), 1.1j))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
