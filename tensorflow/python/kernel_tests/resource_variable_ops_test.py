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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class ResourceVariableOpsTest(test_util.TensorFlowTestCase):

  def testHandleDtypeShapeMatch(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      with self.assertRaises(ValueError):
        resource_variable_ops.create_variable_op(
            handle, constant_op.constant(0.0, dtype=dtypes.float32)).run()
      with self.assertRaises(ValueError):
        resource_variable_ops.create_variable_op(
            handle, constant_op.constant([0], dtype=dtypes.int32)).run()
      resource_variable_ops.create_variable_op(
          handle, constant_op.constant(0, dtype=dtypes.int32)).run()

  def testDtypeSurvivesIdentity(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      id_handle = array_ops.identity(handle)
      resource_variable_ops.create_variable_op(
          id_handle, constant_op.constant(0, dtype=dtypes.int32)).run()


if __name__ == "__main__":
  test.main()
