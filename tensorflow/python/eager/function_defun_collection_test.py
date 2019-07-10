# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DefunCollectionTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='Defun', function_decorator=function.defun),
      dict(
          testcase_name='DefFunction',
          function_decorator=def_function.function))
  def testCollectionValueAccess(self, function_decorator):
    """Read values from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection('x', x)
        ops.add_to_collection('y', y)

        @function_decorator
        def fn():
          x_const = constant_op.constant(ops.get_collection('x')[0])
          y_const = constant_op.constant(ops.get_collection('y')[0])
          z = math_ops.add(x_const, y_const)
          ops.add_to_collection('z', 7)
          return z

        self.assertEqual(7, int(self.evaluate(fn())))
        self.assertEquals(ops.get_collection('x'), [2])
        self.assertEquals(ops.get_collection('y'), [5])
        self.assertEquals(ops.get_collection('z'), [])

  @parameterized.named_parameters(
      dict(testcase_name='Defun', function_decorator=function.defun),
      dict(
          testcase_name='DefFunction',
          function_decorator=def_function.function))
  def testCollectionVariableValueAccess(self, function_decorator):
    """Read variable value from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        v = resource_variable_ops.ResourceVariable(1.0)

        @function_decorator
        def f():
          return v.read_value()

        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(1.0, float(self.evaluate(f())))
        self.assertEquals(
            len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)), 1)

  def testCollectionVariableValueWrite(self):
    """Write variable value inside defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        @function.defun
        def f():
          v = resource_variable_ops.ResourceVariable(2.0)
          return v

        _ = f.get_concrete_function()
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(2.0, float(self.evaluate(f())))
        self.assertEquals(
            len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)), 1)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
