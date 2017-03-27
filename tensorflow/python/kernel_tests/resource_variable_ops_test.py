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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
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

  def testGPU(self):
    with self.test_session(use_gpu=True) as sess:
      abc = variable_scope.get_variable(
          "abc",
          shape=[1],
          initializer=init_ops.ones_initializer(),
          use_resource=True)

      sess.run(variables.global_variables_initializer())
      print(sess.run(abc))

  def testInitFn(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                                 dtype=dtypes.float32)
      self.assertEqual(v.handle.op.colocation_groups(),
                       v.initializer.inputs[1].op.colocation_groups())

  def testInitFnDtype(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                                 dtype=dtypes.float32)
      self.assertEqual(dtypes.float32, v.value().dtype)

  def testInitFnNoDtype(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1)
      self.assertEqual(dtypes.int32, v.value().dtype)

  def testInitializeAllVariables(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.float32)
      with self.assertRaises(errors.NotFoundError):
        v.value().eval()
      variables.global_variables_initializer().run()
      self.assertEqual(1.0, v.value().eval())

  def testOperatorOverload(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      self.assertEqual(2.0, (v+v).eval())

  def testAssignMethod(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      v.assign(2.0).eval()
      self.assertEqual(2.0, v.value().eval())

  def testAssignAddMethod(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      v.assign_add(1.0).eval()
      self.assertEqual(2.0, v.value().eval())

  def testAssignSubMethod(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(3.0)
      variables.global_variables_initializer().run()
      v.assign_sub(1.0).eval()
      self.assertEqual(2.0, v.value().eval())

  def testDestroyResource(self):
    with self.test_session() as sess:
      v = resource_variable_ops.ResourceVariable(3.0)
      variables.global_variables_initializer().run()
      self.assertEqual(3.0, v.value().eval())
      sess.run(resource_variable_ops.destroy_resource_op(v.handle))
      with self.assertRaises(errors.NotFoundError):
        v.value().eval()
      # Handle to a resource not actually created.
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      # Should raise no exception
      sess.run(resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=True))

  def testAssignDifferentShapes(self):
    with self.test_session() as sess, variable_scope.variable_scope(
        "foo", use_resource=True):
      var = variable_scope.get_variable("x", shape=[1, 1], dtype=dtypes.float32)
      placeholder = array_ops.placeholder(dtypes.float32)
      assign = var.assign(placeholder)
      sess.run([assign],
               feed_dict={placeholder: np.zeros(shape=[2, 2],
                                                dtype=np.float32)})

if __name__ == "__main__":
  test.main()
