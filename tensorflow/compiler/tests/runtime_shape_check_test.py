# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for shape checks at runtime in XLA:GPU."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class RuntimeShapeCheckTest(xla_test.XLATestCase):

  def testUniqueDifferentSizes(self):
    """Test that we correctly check for shape mismatches at runtime."""
    if 'tpu' in self.device.lower():
      self.skipTest('We do not check shapes on TPU')

    with ops.device(f'device:{self.device}:0'):

      @def_function.function(jit_compile=True)
      def f(x, y):
        return array_ops.unique(x).y + array_ops.unique(y).y

      f(constant_op.constant([3.1, 3.2]), constant_op.constant([3.3, 3.2]))

      with self.assertRaisesRegex(errors.InternalError, 'different size'):
        f(
            constant_op.constant([3.1, 3.2]),
            constant_op.constant([3.1, 3.2, 3.3]))

  def testWhereOpDifferentSizes(self):
    """Test shape mismatches with multiple dimensions."""
    if 'tpu' in self.device.lower():
      self.skipTest('We do not check shapes on TPU')

    with ops.device(f'device:{self.device}:0'):

      @def_function.function(jit_compile=True)
      def f(x, y):
        return array_ops.where(x) + array_ops.where(y)

      f(
          constant_op.constant([[3.1, 3.2, 0], [3.1, 3.2, 0]]),
          constant_op.constant([[3.3, 3.2, 0, 0, 0], [3.3, 3.2, 0, 0, 0]]))

      with self.assertRaisesRegex(errors.InternalError, 'different size'):
        f(
            constant_op.constant([[3.1, 3.2, 0], [3.1, 3.2, 0]]),
            constant_op.constant([[3.3, 3.2, 0, 0, 0], [3.3, 3.2, 3.3, 0, 0]]))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
