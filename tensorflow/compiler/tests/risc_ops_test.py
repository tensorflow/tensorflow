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
"""Tests for RISC Ops."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.risc import risc_ops
from tensorflow.python.platform import test


class XlaRiscOpsTest(xla_test.XLATestCase):

  def testRiscAddBasic(self):

    @def_function.function(jit_compile=True)
    def f(a, b):
      return risc_ops.risc_add(a, b)

    a = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                             dtype=dtypes.float32)
    b = constant_op.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                             dtype=dtypes.float32)
    self.assertAllEqual(f(a, b), [[8.0, 10.0], [12.0, 14.0], [16.0, 18.0]])

  def testRiscDotBasic(self):

    @def_function.function(jit_compile=True)
    def f(a, b):
      return risc_ops.risc_dot(a, b)

    a = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                             dtype=dtypes.float32)
    b = constant_op.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                             dtype=dtypes.float32)
    self.assertAllEqual(
        f(a, b), [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]])

  def testRiscDotDimensionMismatch(self):

    @def_function.function(jit_compile=True)
    def f(a, b):
      return risc_ops.risc_dot(a, b)

    a = constant_op.constant([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                             dtype=dtypes.float32)
    b = constant_op.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                             dtype=dtypes.float32)
    self.assertRaisesRegex(ValueError, "Dimensions must be equal", f, a, b)

  def testRiscDotTransposeA(self):

    @def_function.function(jit_compile=True)
    def f(a, b):
      return risc_ops.risc_dot(a, b, transpose_a=True)

    a = constant_op.constant([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                             dtype=dtypes.float32)
    b = constant_op.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                             dtype=dtypes.float32)
    self.assertAllEqual(
        f(a, b), [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
