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
"""Tests for VariableSpec."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

VariableSpec = resource_variable_ops.VariableSpec


class VariableSpecTest(test.TestCase, parameterized.TestCase):

  def test_properties(self):
    spec = VariableSpec(shape=(None, None, None))
    self.assertIsNone(spec.name)
    self.assertEqual(spec.shape.as_list(), [None, None, None])
    self.assertEqual(spec.dtype, dtypes.float32)
    self.assertTrue(spec.trainable)
    self.assertIs(spec.value_type, resource_variable_ops.BaseResourceVariable)
    self.assertEqual(spec._component_specs,
                     tensor_spec.TensorSpec(spec.shape, dtypes.resource))

    spec2 = VariableSpec(shape=(1, 2, 3), dtype=dtypes.float64,
                         trainable=False)
    self.assertEqual(spec2.shape.as_list(), [1, 2, 3])
    self.assertEqual(spec2.dtype, dtypes.float64)
    self.assertFalse(spec2.trainable)
    self.assertIs(spec2.value_type, resource_variable_ops.BaseResourceVariable)
    self.assertEqual(spec2._component_specs,
                     tensor_spec.TensorSpec(spec2.shape, dtypes.resource))

  def test_compatibility(self):
    spec = VariableSpec(shape=None)
    spec2 = VariableSpec(shape=[None, 15])
    spec3 = VariableSpec(shape=[None])

    self.assertTrue(spec.is_compatible_with(spec2))
    self.assertFalse(spec2.is_compatible_with(spec3))

    var = resource_variable_ops.UninitializedVariable(
        shape=[3, 15], dtype=dtypes.float32)
    var2 = resource_variable_ops.UninitializedVariable(
        shape=[3], dtype=dtypes.int32)
    var3 = resource_variable_ops.ResourceVariable(
        initial_value=np.ones((3, 15), dtype=np.float32))
    var4 = resource_variable_ops.ResourceVariable(
        initial_value=np.ones((3,), dtype=np.int32))

    self.assertTrue(spec.is_compatible_with(var))
    self.assertFalse(spec2.is_compatible_with(var2))
    self.assertTrue(spec.is_compatible_with(var3))
    self.assertFalse(spec2.is_compatible_with(var4))

    spec4 = VariableSpec(shape=None, dtype=dtypes.int32)
    spec5 = VariableSpec(shape=[None], dtype=dtypes.int32)

    self.assertFalse(spec.is_compatible_with(spec4))
    self.assertTrue(spec4.is_compatible_with(spec5))
    self.assertTrue(spec4.is_compatible_with(var2))
    self.assertTrue(spec4.is_compatible_with(var4))

    tensor = constant_op.constant([1, 2, 3])
    self.assertFalse(spec4.is_compatible_with(tensor))

  @parameterized.parameters([
      dict(
          initial_value=[1, 2, 3],
          shape=[3],
          dtype=dtypes.int32,
          trainable=False),
      dict(
          initial_value=[[1., 2., 3.]],
          shape=[1, None]),
      dict(
          shape=None),
  ])
  def testFromValue(self,
                    initial_value=None,
                    shape=None,
                    dtype=dtypes.float32,
                    trainable=True):
    if initial_value is None:
      var = resource_variable_ops.UninitializedVariable(
          shape=shape, dtype=dtype, trainable=trainable)
    else:
      var = resource_variable_ops.ResourceVariable(
          initial_value=initial_value,
          shape=shape,
          dtype=dtype,
          trainable=trainable)
    spec = resource_variable_ops.VariableSpec.from_value(var)
    expected_spec = resource_variable_ops.VariableSpec(
        shape=shape, dtype=dtype, trainable=trainable)
    self.assertEqual(spec, expected_spec)

  @parameterized.parameters([
      dict(
          initial_value=[1., 2., 3.],
          shape=[3]),
      dict(
          initial_value=[1., 2., 3.],
          shape=None),
      dict(
          initial_value=[[1, 2, 3]],
          shape=[1, None],
          dtype=dtypes.int32),
      dict(
          initial_value=[[1, 2, 3]],
          shape=[None, None],
          dtype=dtypes.int32),
      dict(
          shape=[3]),
      dict(
          shape=[None]),
      dict(
          shape=None,
          dtype=dtypes.int32),
  ])
  def testToFromComponents(self,
                           initial_value=None,
                           shape=None,
                           dtype=dtypes.float32,
                           trainable=True):
    if not context.executing_eagerly():
      return

    if initial_value is None:
      var = resource_variable_ops.UninitializedVariable(
          shape=shape,
          dtype=dtype,
          trainable=trainable)
    else:
      var = resource_variable_ops.ResourceVariable(
          initial_value=initial_value,
          shape=shape,
          dtype=dtype,
          trainable=trainable)
    spec = resource_variable_ops.VariableSpec.from_value(var)
    components = spec._to_components(var)
    rebuilt_var = spec._from_components(components)
    self.assertAllEqual(rebuilt_var, var)
    self.assertEqual(rebuilt_var.trainable, trainable)

  def testSerialize(self):
    shape = [1, 3]
    dtype = dtypes.int32
    trainable = False
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable)
    serialization = spec._serialize()
    expected_serialization = (shape, dtype, trainable)
    self.assertEqual(serialization, expected_serialization)
    rebuilt_spec = spec._deserialize(serialization)
    self.assertEqual(rebuilt_spec, spec)

  def testRepr(self):
    shape = (1, 3)
    dtype = dtypes.int32
    trainable = False
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable)
    spec_repr = repr(spec)
    expected_repr = (
        f"VariableSpec(shape={shape}, dtype={dtype}, trainable={trainable})")
    self.assertEqual(spec_repr, expected_repr)

  def testHash(self):
    shape = (1, 3)
    dtype = dtypes.int32
    trainable = False
    spec = resource_variable_ops.VariableSpec(shape, dtype, trainable)
    spec_hash = hash(spec)
    expected_hash = hash((shape, dtype, trainable))
    self.assertEqual(spec_hash, expected_hash)

  def testEquality(self):
    spec = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    spec2 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    self.assertEqual(spec2, spec)
    spec3 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, False)
    self.assertEqual(spec3, spec)
    spec4 = resource_variable_ops.VariableSpec([1, 3], dtypes.float32, True)
    self.assertNotEqual(spec4, spec)
    spec5 = resource_variable_ops.VariableSpec([3, 3], dtypes.float32, True)
    self.assertNotEqual(spec5, spec)
    spec6 = resource_variable_ops.VariableSpec([1, 3], dtypes.int32, True)
    self.assertNotEqual(spec6, spec)


if __name__ == "__main__":
  test.main()
