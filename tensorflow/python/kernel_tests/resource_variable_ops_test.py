# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.resource_variable_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class ResourceVariableOpsTest(test_util.TensorFlowTestCase):

  def testHandleDtypeShapeMatch(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(0.0, dtype=dtypes.float32)).run()
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([0], dtype=dtypes.int32)).run()
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(0, dtype=dtypes.int32)).run()

  def testDtypeSurvivesIdentity(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      id_handle = array_ops.identity(handle)
      resource_variable_ops.assign_variable_op(
          id_handle, constant_op.constant(0, dtype=dtypes.int32)).run()

  def testCreateRead(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(1, dtype=dtypes.int32)).run()
      value = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32).eval()
      self.assertAllEqual(1, value)

  def testManyAssigns(self):
    with self.test_session() as session:
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      create = resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(1, dtype=dtypes.int32))
      with ops.control_dependencies([create]):
        first_read = resource_variable_ops.read_variable_op(
            handle, dtype=dtypes.int32)
      with ops.control_dependencies([first_read]):
        write = resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(2, dtype=dtypes.int32))
      with ops.control_dependencies([write]):
        second_read = resource_variable_ops.read_variable_op(
            handle, dtype=dtypes.int32)
      f, s = session.run([first_read, second_read])
      self.assertEqual(f, 1)
      self.assertEqual(s, 2)

  def testAssignAdd(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(1, dtype=dtypes.int32)).run()
      resource_variable_ops.assign_add_variable_op(
          handle, constant_op.constant(1, dtype=dtypes.int32)).run()
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(read.eval(), 2)

  def testScatterAdd(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([[1]], dtype=dtypes.int32)).run()
      resource_variable_ops.resource_scatter_add(
          handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)).run()
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(read.eval(), [[3]])


if __name__ == "__main__":
  test.main()
