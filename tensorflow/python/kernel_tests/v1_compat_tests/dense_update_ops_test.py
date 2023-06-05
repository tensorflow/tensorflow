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
"""Tests for dense_update_ops that only work in V1."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test as test_lib


class AssignOpTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("Non-strict shape assignment only in V1.")
  # Bypassing Shape validation in assign with validate_shape only works in V1.
  # TF V2 raises error:
  # ValueError: Cannot assign value to variable ' Variable:0': Shape mismatch.
  # The variable shape (1,), and the assigned value shape (1024, 1024) are
  # incompatible.
  # Thus, test is enabled only for V1.
  def testAssignNonStrictShapeChecking(self):
    data = array_ops.fill([1024, 1024], 0)
    p = variable_v1.VariableV1([1])
    a = state_ops.assign(p, data, validate_shape=False)
    self.evaluate(a)
    self.assertAllEqual(p, self.evaluate(data))

    # Assign to yet another shape
    data2 = array_ops.fill([10, 10], 1)
    a2 = state_ops.assign(p, data2, validate_shape=False)
    self.evaluate(a2)
    self.assertAllEqual(p, self.evaluate(data2))

  @test_util.run_v1_only("Variables need initialization only in V1,")
  def testInitRequiredAssignAdd(self):
    p = variable_v1.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
    a = state_ops.assign_add(p, array_ops.fill([1024, 1024], 0))
    with self.assertRaisesOpError("use uninitialized"):
      self.evaluate(a)

  @test_util.run_v1_only("Variables need initialization only in V1.")
  def testInitRequiredAssignSub(self):
    p = variable_v1.VariableV1(array_ops.fill([1024, 1024], 1), dtypes.int32)
    a = state_ops.assign_sub(p, array_ops.fill([1024, 1024], 0))
    with self.assertRaisesOpError("use uninitialized"):
      self.evaluate(a)


if __name__ == "__main__":
  test_lib.main()
