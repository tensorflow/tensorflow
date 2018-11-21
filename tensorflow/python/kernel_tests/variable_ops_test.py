# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.variable_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_NP_TO_TF = {
    np.float32: dtypes.float32,
    np.float64: dtypes.float64,
    np.int32: dtypes.int32,
    np.int64: dtypes.int64,
}


class VariableOpTest(test.TestCase):

  def _initFetch(self, x, tftype, use_gpu=None):
    with self.test_session(use_gpu=use_gpu):
      p = state_ops.variable_op(x.shape, tftype)
      op = state_ops.assign(p, x)
      op.op.run()
      return self.evaluate(p)

  def _testTypes(self, vals):
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
      self.setUp()
      x = vals.astype(dtype)
      tftype = _NP_TO_TF[dtype]
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=False))
      # NOTE(touts): the GPU test should pass for all types, whether the
      # Variable op has an implementation for that type on GPU as we expect
      # that Variable and Assign have GPU implementations for matching tf.
      self.assertAllEqual(x, self._initFetch(x, tftype, use_gpu=True))

  def testBasic(self):
    self._testTypes(np.arange(0, 20).reshape([4, 5]))

  def testset_shape(self):
    p = state_ops.variable_op([1, 2], dtypes.float32)
    self.assertEqual([1, 2], p.get_shape())
    p = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), p.get_shape())

  def testAssign(self):
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32)
    self.assertShapeEqual(value, var)
    assigned = state_ops.assign(var, value)
    self.assertShapeEqual(value, assigned)

  def testAssignNoValidateShape(self):
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32)
    self.assertShapeEqual(value, var)
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertShapeEqual(value, assigned)

  def testAssignNoVarShape(self):
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
    assigned = state_ops.assign(var, value)
    self.assertShapeEqual(value, assigned)

  def testAssignNoVarShapeNoValidateShape(self):
    value = np.array([[42.0, 43.0]])
    var = state_ops.variable_op(value.shape, dtypes.float32, set_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertShapeEqual(value, assigned)

  def _NewShapelessTensor(self):
    tensor = array_ops.placeholder(dtypes.float32)
    self.assertEqual(tensor_shape.unknown_shape(), tensor.get_shape())
    return tensor

  def testAssignNoValueShape(self):
    value = self._NewShapelessTensor()
    shape = [1, 2]
    var = state_ops.variable_op(shape, dtypes.float32)
    assigned = state_ops.assign(var, value)
    self.assertEqual(shape, var.get_shape())
    self.assertEqual(shape, assigned.get_shape())

  def testAssignNoValueShapeNoValidateShape(self):
    value = self._NewShapelessTensor()
    shape = [1, 2]
    var = state_ops.variable_op(shape, dtypes.float32)
    self.assertEqual(shape, var.get_shape())
    assigned = state_ops.assign(var, value, validate_shape=False)
    self.assertEqual(tensor_shape.unknown_shape(), assigned.get_shape())

  def testAssignNoShape(self):
    with self.cached_session():
      value = self._NewShapelessTensor()
      var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
      self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
      self.assertEqual(tensor_shape.unknown_shape(),
                       state_ops.assign(var, value).get_shape())

  def testAssignNoShapeNoValidateShape(self):
    with self.cached_session():
      value = self._NewShapelessTensor()
      var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
      self.assertEqual(tensor_shape.unknown_shape(), var.get_shape())
      self.assertEqual(
          tensor_shape.unknown_shape(),
          state_ops.assign(
              var, value, validate_shape=False).get_shape())

  def testAssignUpdate(self):
    var = state_ops.variable_op([1, 2], dtypes.float32)
    added = state_ops.assign_add(var, [[2.0, 3.0]])
    self.assertEqual([1, 2], added.get_shape())
    subbed = state_ops.assign_sub(var, [[12.0, 13.0]])
    self.assertEqual([1, 2], subbed.get_shape())

  def testAssignUpdateNoVarShape(self):
    var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    added = state_ops.assign_add(var, [[2.0, 3.0]])
    self.assertEqual([1, 2], added.get_shape())
    subbed = state_ops.assign_sub(var, [[12.0, 13.0]])
    self.assertEqual([1, 2], subbed.get_shape())

  def testAssignUpdateNoValueShape(self):
    var = state_ops.variable_op([1, 2], dtypes.float32)
    added = state_ops.assign_add(var, self._NewShapelessTensor())
    self.assertEqual([1, 2], added.get_shape())
    subbed = state_ops.assign_sub(var, self._NewShapelessTensor())
    self.assertEqual([1, 2], subbed.get_shape())

  def testAssignUpdateNoShape(self):
    var = state_ops.variable_op([1, 2], dtypes.float32, set_shape=False)
    added = state_ops.assign_add(var, self._NewShapelessTensor())
    self.assertEqual(tensor_shape.unknown_shape(), added.get_shape())
    subbed = state_ops.assign_sub(var, self._NewShapelessTensor())
    self.assertEqual(tensor_shape.unknown_shape(), subbed.get_shape())

  def testTemporaryVariable(self):
    with self.test_session(use_gpu=True):
      var = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="foo")
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = gen_state_ops.destroy_temporary_variable(var, var_name="foo")
      self.assertAllClose([[10.0, 12.0]], self.evaluate(final))

  def testDestroyNonexistentTemporaryVariable(self):
    with self.test_session(use_gpu=True):
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      final = gen_state_ops.destroy_temporary_variable(var, var_name="bad")
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  def testDuplicateTemporaryVariable(self):
    with self.test_session(use_gpu=True):
      var1 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="dup")
      var1 = state_ops.assign(var1, [[1.0, 2.0]])
      var2 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="dup")
      var2 = state_ops.assign(var2, [[3.0, 4.0]])
      final = var1 + var2
      with self.assertRaises(errors.AlreadyExistsError):
        self.evaluate(final)

  def testDestroyTemporaryVariableTwice(self):
    with self.test_session(use_gpu=True):
      var = gen_state_ops.temporary_variable([1, 2], dtypes.float32)
      val1 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      val2 = gen_state_ops.destroy_temporary_variable(var, var_name="dup")
      final = val1 + val2
      with self.assertRaises(errors.NotFoundError):
        self.evaluate(final)

  def testTemporaryVariableNoLeak(self):
    with self.test_session(use_gpu=True):
      var = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="bar")
      final = array_ops.identity(var)
      self.evaluate(final)

  def testTwoTemporaryVariablesNoLeaks(self):
    with self.test_session(use_gpu=True):
      var1 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="var1")
      var2 = gen_state_ops.temporary_variable(
          [1, 2], dtypes.float32, var_name="var2")
      final = var1 + var2
      self.evaluate(final)

  def testAssignDependencyAcrossDevices(self):
    with self.test_session(use_gpu=True):
      # The variable and an op to increment it are on the GPU.
      var = state_ops.variable_op([1], dtypes.float32)
      state_ops.assign(var, [1.0]).eval()
      increment = state_ops.assign_add(var, [1.0])
      with ops.control_dependencies([increment]):
        with ops.device("/cpu:0"):
          # This mul op is pinned to the CPU, but reads the variable from the
          # GPU. The test ensures that the dependency on 'increment' is still
          # honored, i.e., the Send and Recv from GPU to CPU should take place
          # only after the increment.
          result = math_ops.multiply(var, var)
      self.assertAllClose([4.0], self.evaluate(result))

  def testIsVariableInitialized(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        v0 = state_ops.variable_op([1, 2], dtypes.float32)
        self.assertEqual(False, variables.is_variable_initialized(v0).eval())
        state_ops.assign(v0, [[2.0, 3.0]]).eval()
        self.assertEqual(True, variables.is_variable_initialized(v0).eval())


if __name__ == "__main__":
  test.main()
