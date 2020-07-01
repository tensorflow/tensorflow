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
"""Functional test for slot_creator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import slot_creator


class SlotCreatorTest(test.TestCase):

  @test_util.run_v1_only("b/120545219")
  def testCreateSlotFromVariable(self):
    with self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      slot = slot_creator.create_slot(v, v.initialized_value(), name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([1.0, 2.5], self.evaluate(slot))

  @test_util.run_deprecated_v1
  def testCreateSlotFromTensor(self):
    with self.cached_session():
      v = constant_op.constant([1.0, 2.5], name="const")
      slot = slot_creator.create_slot(v, v * 2, name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([2.0, 5.0], self.evaluate(slot))

  @test_util.run_deprecated_v1
  def testCreateZerosSlotFromVariable(self):
    with self.cached_session():
      v = variables.Variable([1.0, 2.5], name="var")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  @test_util.run_v1_only("b/120545219")
  def testCreateZerosSlotFromDynamicShapedVariable(self):
    with self.cached_session():
      dyn_shape = constant_op.constant([2], dtype=dtypes.int32)
      dyn_shape = array_ops.placeholder_with_default(dyn_shape,
                                                     shape=[None])
      v = variable_scope.get_variable(
          "var",
          initializer=random_ops.random_uniform(dyn_shape,
                                                dtype=dtypes.float64),
          validate_shape=False)
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("var/slot", slot.op.name)
      self.assertEqual([2], array_ops.shape(slot).eval())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  @test_util.run_deprecated_v1
  def testCreateZerosSlotFromTensor(self):
    with self.cached_session():
      v = constant_op.constant([1.0, 2.5], name="const")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(v, name="slot")

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], slot.get_shape().as_list())
      self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  @test_util.run_deprecated_v1
  def testCreateZerosSlotFromDynamicShapedTensor(self):
    with self.cached_session():
      v = random_ops.random_uniform([2], dtype=dtypes.float64)
      v = array_ops.placeholder_with_default(v, shape=[None], name="const")
      with ops.control_dependencies(None):
        slot = slot_creator.create_zeros_slot(
            v, name="slot", dtype=dtypes.float64)

      self.evaluate(variables.global_variables_initializer())

      self.assertEqual("const/slot", slot.op.name)
      self.assertEqual([2], array_ops.shape(slot).eval())
      self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
      self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

  @test_util.run_v1_only("b/120545219")
  def testCreateSlotFromVariableRespectsScope(self):
    # See discussion on #2740.
    with self.cached_session():
      with variable_scope.variable_scope("scope"):
        v = variables.Variable([1.0, 2.5], name="var")
        slot = slot_creator.create_slot(v, v.initialized_value(), name="slot")
        self.assertEqual("scope/scope/var/slot", slot.op.name)

  @test_util.run_deprecated_v1
  def testCreateSlotFromFirstMDimensionVariable(self):
    with self.test_session():
      s = variables.Variable([1.0, 2.5], name="var")
      p_v = variable_scope.get_variable(
          "var",
          shape=[2, 2],
          partitioner=partitioned_variables.fixed_size_partitioner(2))
      for i, v in enumerate(p_v):
        slot = slot_creator.create_slot(v, s.initialized_value(), name="slot")
        si = slot._save_slice_info

        variables.global_variables_initializer().run()

        self.assertEqual("var/part_%d/slot" % i, slot.op.name)
        self.assertEqual([2], slot.get_shape().as_list())
        self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
        self.assertAllEqual([1.0, 2.5], slot)
        self.assertAllEqual([2], si.full_shape)
        self.assertAllEqual([i], si.var_offset)
        self.assertAllEqual([1], si.var_shape)

  @test_util.run_deprecated_v1
  def testCreateSlotFromScalarVariable(self):
    with self.test_session():
      s = variables.Variable(1.0, name="var")
      p_v = variable_scope.get_variable(
          "var",
          shape=[2, 2],
          partitioner=partitioned_variables.fixed_size_partitioner(2))
      for i, v in enumerate(p_v):
        slot = slot_creator.create_slot(v, s.initialized_value(), name="slot")

        variables.global_variables_initializer().run()

        self.assertEqual("var/part_%d/slot" % i, slot.op.name)
        self.assertEqual([], slot.get_shape().as_list())
        self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
        self.assertAllEqual(1.0, slot)


if __name__ == "__main__":
  test.main()
