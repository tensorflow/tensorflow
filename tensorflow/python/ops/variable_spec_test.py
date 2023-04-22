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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

VariableSpec = resource_variable_ops.VariableSpec


class VariableSpecTest(test.TestCase):

  def test_properties(self):
    spec = VariableSpec(shape=(1, 2, 3), dtype=dtypes.float64, name='vs',
                        trainable=True)
    self.assertEqual('vs', spec.name)
    self.assertEqual(tensor_shape.TensorShape((1, 2, 3)), spec.shape)
    self.assertEqual(dtypes.float64, spec.dtype)
    self.assertEqual(True, spec.trainable)

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

    self.assertTrue(spec2.is_compatible_with(var))
    self.assertFalse(spec3.is_compatible_with(var2))

    spec4 = VariableSpec(shape=None, dtype=dtypes.int32)
    spec5 = VariableSpec(shape=[None], dtype=dtypes.int32)

    self.assertFalse(spec.is_compatible_with(spec4))
    self.assertTrue(spec4.is_compatible_with(spec5))
    self.assertTrue(spec4.is_compatible_with(var2))

    tensor = constant_op.constant([1, 2, 3])
    self.assertFalse(spec4.is_compatible_with(tensor))


if __name__ == '__main__':
  test.main()
