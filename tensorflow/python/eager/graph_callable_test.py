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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import graph_callable
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class GraphCallableTest(test.TestCase):

  def testBasic(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      return v + x

    self.assertEqual(
        2, my_function(constant_op.constant(2, dtype=dtypes.float32)).numpy())

    my_function.variables[0].assign(1.)
    self.assertEqual(
        3, my_function(constant_op.constant(2, dtype=dtypes.float32)).numpy())

  def testFunctionWithoutReturnValue(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      v.assign(x)

    my_function(constant_op.constant(4, dtype=dtypes.float32))
    self.assertAllEqual(4, my_function.variables[0].read_value())

  def testFunctionWithoutReturnValueAndArgs(self):

    @graph_callable.graph_callable([])
    def my_function():
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      v.assign(4)

    my_function()
    self.assertAllEqual(4, my_function.variables[0].read_value())

  def testVariableAPI(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      return v.read_value() + x

    self.assertEqual(
        2, my_function(constant_op.constant(2, dtype=dtypes.float32)).numpy())

    my_function.variables[0].assign(1.)
    self.assertEqual(
        3, my_function(constant_op.constant(2, dtype=dtypes.float32)).numpy())

  def testTensorShape(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(1), dtype=dtypes.float32)])
    def my_function(x):
      _ = x.get_shape()
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=[x.shape[0]])
      self.assertEqual(v.shape[0], x.shape[0])
      return v + x

    self.assertEqual([2.],
                     my_function(
                         constant_op.constant([2.],
                                              dtype=dtypes.float32)).numpy())

  def testUpdatesAreOrdered(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      v.assign(x + 1)
      v.assign(v * x)
      return v.read_value()

    self.assertAllEqual(my_function(constant_op.constant(2.0)), 6.0)

  def testEmptyInitializer(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(1), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable("v", shape=[1])
      return x + 0 * v

    self.assertEqual([2.],
                     my_function(
                         constant_op.constant([2.],
                                              dtype=dtypes.float32)).numpy())

  def testMismatchingNumArgs(self):
    # pylint: disable=anomalous-backslash-in-string
    with self.assertRaisesRegexp(TypeError,
                                 "The number of arguments accepted by the "
                                 "decorated function `my_function` \(2\) must "
                                 "match the number of ShapeAndDtype objects "
                                 "passed to the graph_callable\(\) decorator "
                                 "\(1\)."):
      @graph_callable.graph_callable([
          graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
      def my_function(x, y):  # pylint: disable=unused-variable
        return x + y
    # pylint: enable=anomalous-backslash-in-string

  def testPureFunction(self):

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    self.assertAllEqual(5, f(constant_op.constant(2)))

  def testNestedFunction(self):
    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    # A graph_callable that will invoke the TensorFlow function.
    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_one(x):
      return add(x, 1)

    self.assertAllEqual(3, add_one(constant_op.constant(2)))

  # TODO(ashankar): Make this work.
  # The problem is that the two graph_callables (for add_one and add_two)
  # are both trying to register the FunctionDef corresponding to "add".
  def DISABLED_testRepeatedUseOfSubFunction(self):

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_one(x):
      return add(x, 1)

    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.int32)])
    def add_two(x):
      return add(x, 2)

    two = constant_op.constant(2)
    self.assertAllEqual(3, add_one(two))
    self.assertAllEqual(4, add_two(two))

  def testNestedSequenceInputs(self):
    sd = graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)
    @graph_callable.graph_callable([[sd, tuple([sd, sd]), sd]])
    def my_op(inputs):
      a, b, c = inputs
      e, f = b
      v = variable_scope.get_variable(
          "my_v", initializer=init_ops.zeros_initializer(), shape=())
      return [a + a + v, tuple([e + e, f + f]), c + c], a + e + f + c + v

    inputs = [constant_op.constant(1.),
              [constant_op.constant(2.), constant_op.constant(3.)],
              constant_op.constant(4.)]
    ret = my_op(inputs)
    self.assertEqual(len(ret), 2.)
    self.assertAllEqual(ret[1], 10.)

    my_op.variables[0].assign(1.)
    ret = my_op(inputs)
    self.assertAllEqual(ret[1], 11.)

  def testVariableShapeIsTensorShape(self):
    @graph_callable.graph_callable([])
    def my_function():
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      self.assertIsInstance(v.get_shape(), tensor_shape.TensorShape)

    my_function()

  def testIncorrectlyShapedInputs(self):
    @graph_callable.graph_callable(
        [graph_callable.ShapeAndDtype(shape=(3), dtype=dtypes.float32)])
    def my_function(x):
      v = variable_scope.get_variable(
          "v", initializer=init_ops.zeros_initializer(), shape=())
      return v + x

    with self.assertRaises(ValueError):
      my_function([1, 2])

    self.assertTrue(([1, 2, 3] == my_function(
        constant_op.constant([1, 2, 3], dtype=dtypes.float32)).numpy()).all())

  def testGradients(self):
    @graph_callable.graph_callable([])
    def my_function():
      v = variable_scope.get_variable(
          "v", initializer=init_ops.constant_initializer(3.), shape=())
      return v * v

    grad_fn = backprop.implicit_grad(my_function)
    grads_and_vars = list(zip(*grad_fn()))
    self.assertAllEqual(6., grads_and_vars[0][0])


if __name__ == "__main__":
  test.main()
