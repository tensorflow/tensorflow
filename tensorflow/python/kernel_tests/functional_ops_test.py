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
"""Tests for tensorflow.kernels.functional_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util import compat


# pylint: disable=invalid-name
def simple_scoped_fn(a, x):
  """Simple function: (a, x) -> 2(x+a), but with "2" as a variable in scope."""
  with variable_scope.variable_scope("body"):
    # Dummy variable, just to check that scoping works as intended.
    two = variable_scope.get_variable(
        "two", [],
        dtype=dtypes.int32,
        initializer=init_ops.constant_initializer(2))
    return math_ops.multiply(math_ops.add(a, x), two)


class FunctionalOpsTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_Simple(self):
    with self.test_session():
      elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

      r = functional_ops.foldl(
          lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
          elems)
      self.assertAllEqual(208, self.evaluate(r))

      r = functional_ops.foldl(
          lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
          elems,
          initializer=10)
      self.assertAllEqual(880, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_SingleInputMultiOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array([1, -1.0])
      r = functional_ops.foldl(lambda a, x: a + x, elems, initializer)
      r_value = self.evaluate(r)

      self.assertAllEqual(22, r_value[0])
      self.assertAllEqual(20, r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testFoldl_MultiInputSingleOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      r = functional_ops.foldl(lambda a, x: a + x[0] + x[1], (elems, -elems),
                               initializer)
      self.assertAllEqual(1, self.evaluate(r))

  def testFoldl_Scoped(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.foldl(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(208, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.foldl(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(880, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_Simple(self):
    with self.test_session():
      elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

      r = functional_ops.foldr(
          lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
          elems)
      self.assertAllEqual(450, self.evaluate(r))

      r = functional_ops.foldr(
          lambda a, x: math_ops.multiply(math_ops.add(a, x), 2),
          elems,
          initializer=10)
      self.assertAllEqual(1282, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_SingleInputMultiOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array([1, -1.0])
      r = functional_ops.foldr(lambda a, x: a + x, elems, initializer)
      r_value = self.evaluate(r)

      self.assertAllEqual(22, r_value[0])
      self.assertAllEqual(20, r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testFoldr_MultiInputSingleOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      r = functional_ops.foldr(lambda a, x: a + x[0] + x[1], (elems, -elems),
                               initializer)
      self.assertAllEqual(1, self.evaluate(r))

  def testFoldr_Scoped(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.foldr(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(450, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.foldr(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(1282, self.evaluate(r))

  # pylint: disable=unnecessary-lambda
  def testFold_Grad(self):
    with self.test_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")
      r = functional_ops.foldl(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(720.0, self.evaluate(r))

      r = functional_ops.foldr(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(720.0, self.evaluate(r))
  # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testMap_Simple(self):
    with self.test_session():
      nums = [1, 2, 3, 4, 5, 6]
      elems = constant_op.constant(nums, name="data")
      r = functional_ops.map_fn(
          lambda x: math_ops.multiply(math_ops.add(x, 3), 2), elems)
      self.assertAllEqual(
          np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  def testMapSparseTensor(self):
    with self.test_session():
      with self.assertRaises(TypeError):
        functional_ops.map_fn(
            lambda x: x,
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1], [1, 0]],
                values=constant_op.constant([0, 1, 2]),
                dense_shape=[2, 2]))

  @test_util.run_in_graph_and_eager_modes
  def testMapOverScalarErrors(self):
    with self.assertRaisesRegexp(ValueError, "not scalars"):
      functional_ops.map_fn(lambda x: x, [1, 2])
    with self.assertRaisesRegexp(ValueError, "not a scalar"):
      functional_ops.map_fn(lambda x: x, 1)

  def testMap_Scoped(self):
    with self.test_session() as sess:

      def double_scoped(x):
        """2x with a dummy 2 that is scoped."""
        with variable_scope.variable_scope("body"):
          # Dummy variable, just to check that scoping works as intended.
          two = variable_scope.get_variable(
              "two", [],
              dtype=dtypes.int32,
              initializer=init_ops.constant_initializer(2))
          return math_ops.multiply(x, two)

      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
        doubles = np.array([2 * x for x in [1, 2, 3, 4, 5, 6]])

        r = functional_ops.map_fn(double_scoped, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(doubles, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.map_fn(double_scoped, elems)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(doubles, self.evaluate(r))

  def testMap_Grad(self):
    with self.test_session():
      param = constant_op.constant(2.0)
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
      y = functional_ops.map_fn(
          lambda x: math_ops.multiply(math_ops.square(x), param), elems)
      r = gradients_impl.gradients(y, param)[0]
      self.assertAllEqual(91.0, self.evaluate(r))
      r = gradients_impl.gradients(y, elems)[0]
      self.assertAllEqual([4.0, 8.0, 12.0, 16.0, 20.0, 24.0], self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SimpleNotTensor(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(
          lambda x: math_ops.multiply(math_ops.add(x, 3), 2), nums)
      self.assertAllEqual(
          np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SingleInputMultiOutput(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(
          lambda x: ((x + 3) * 2, -(x + 3) * 2),
          nums,
          dtype=(dtypes.int64, dtypes.int64))
      self.assertEqual(2, len(r))
      self.assertEqual((6,), r[0].get_shape())
      self.assertEqual((6,), r[1].get_shape())
      received = self.evaluate(r)
      self.assertAllEqual((nums + 3) * 2, received[0])
      self.assertAllEqual(-(nums + 3) * 2, received[1])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiOutputMismatchedDtype(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      with self.assertRaisesRegexp(
          TypeError, r"two structures don't have the same nested structure"):
        # lambda emits tuple, but dtype is a list
        functional_ops.map_fn(
            lambda x: ((x + 3) * 2, -(x + 3) * 2),
            nums,
            dtype=[dtypes.int64, dtypes.int64])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSingleOutput(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(
          lambda x: x[0] * x[1][0] + x[1][1], (nums, (nums, -nums)),
          dtype=dtypes.int64)
      self.assertEqual((6,), r.get_shape())
      received = self.evaluate(r)
      self.assertAllEqual(nums * nums + (-nums), received)

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSameStructureOutput(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(lambda x: (x[1][0], (x[1][1], x[0])),
                                (nums, (2 * nums, -nums)))
      r = [r[0], r[1][0], r[1][1]]
      self.assertEqual((6,), r[0].get_shape())
      self.assertEqual((6,), r[1].get_shape())
      self.assertEqual((6,), r[2].get_shape())
      received = self.evaluate(r)
      self.assertAllEqual(2 * nums, received[0])
      self.assertAllEqual(-nums, received[1])
      self.assertAllEqual(nums, received[2])

  @test_util.run_in_graph_and_eager_modes
  def testScan_Simple(self):
    with self.test_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")

      # pylint: disable=unnecessary-lambda
      r = functional_ops.scan(lambda a, x: math_ops.multiply(a, x), elems)
      self.assertAllEqual([1., 2., 6., 24., 120., 720.], self.evaluate(r))

      r = functional_ops.scan(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      self.assertAllEqual([2., 4., 12., 48., 240., 1440.], self.evaluate(r))
      # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testScan_Reverse(self):
    with self.test_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")

      # pylint: disable=unnecessary-lambda
      r = functional_ops.scan(lambda a, x: math_ops.multiply(a, x), elems,
                              reverse=True)
      self.assertAllEqual([720., 720., 360., 120., 30., 6.], self.evaluate(r))
      r = functional_ops.scan(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v,
          reverse=True)
      self.assertAllEqual([1440., 1440., 720., 240., 60., 12.],
                          self.evaluate(r))
      # pylint: enable=unnecessary-lambda

  @test_util.run_in_graph_and_eager_modes
  def testScan_SingleInputMultiOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = (np.array(1.0), np.array(-1.0))
      r = functional_ops.scan(lambda a, x: (a[0] * x, -a[1] * x), elems,
                              initializer)
      r_value = self.evaluate(r)

      self.assertAllEqual([1.0, 2.0, 6.0, 24.0, 120.0, 720.0], r_value[0])
      self.assertAllEqual([1.0, -2.0, 6.0, -24.0, 120.0, -720.0], r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiInputSingleOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      r = functional_ops.scan(lambda a, x: a * (x[0] + x[1]),
                              (elems + 1, -elems), initializer)
      self.assertAllEqual([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiInputSameTypeOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      r = functional_ops.scan(lambda a, x: (a[0] + x[0], a[1] + x[1]),
                              (elems, -elems))
      r_value = self.evaluate(r)
      self.assertAllEqual(np.cumsum(elems), r_value[0])
      self.assertAllEqual(np.cumsum(-elems), r_value[1])

  @test_util.run_in_graph_and_eager_modes
  def testScan_MultiOutputMismatchedInitializer(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      with self.assertRaisesRegexp(
          ValueError, "two structures don't have the same nested structure"):
        functional_ops.scan(lambda a, x: (a, -a), elems, initializer)

  def testScan_Scoped(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")

        r = functional_ops.scan(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        results = np.array([1, 6, 18, 44, 98, 208])
        self.assertAllEqual(results, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = functional_ops.scan(simple_scoped_fn, elems, initializer=2)
        self.assertEqual(len(variables.trainable_variables()), 1)
        results = np.array([6, 16, 38, 84, 178, 368])
        self.assertAllEqual(results, self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testScanFoldl_Nested(self):
    with self.test_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0], name="data")
      inner_elems = constant_op.constant([0.5, 0.5], name="data")

      def r_inner(a, x):
        return functional_ops.foldl(
            lambda b, y: b * y * x, inner_elems, initializer=a)

      r = functional_ops.scan(r_inner, elems)

      # t == 0 (returns 1)
      # t == 1, a == 1, x == 2 (returns 1)
      #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1
      #   t_1 == 1, b == 1,      y == 0.5, returns b * y * x = 1
      # t == 2, a == 1, x == 3 (returns 1.5*1.5 == 2.25)
      #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1.5
      #   t_1 == 1, b == 1.5,    y == 0.5, returns b * y * x = 1.5*1.5
      # t == 3, a == 2.25, x == 4 (returns 9)
      #   t_0 == 0, b == a == 2.25, y == 0.5, returns b * y * x = 4.5
      #   t_1 == 1, b == 4.5,       y == 0.5, returns b * y * x = 9
      self.assertAllClose([1., 1., 2.25, 9.], self.evaluate(r))

  def testScan_Control(self):
    with self.test_session() as sess:
      s = array_ops.placeholder(dtypes.float32, shape=[None])
      b = array_ops.placeholder(dtypes.bool)

      with ops.control_dependencies([b]):
        c = functional_ops.scan(lambda a, x: x * a, s)
      self.assertAllClose(
          np.array([1.0, 3.0, 9.0]), sess.run(c, {s: [1, 3, 3],
                                                  b: True}))

  def testScan_Grad(self):
    with self.test_session():
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = constant_op.constant(2.0, name="v")

      # pylint: disable=unnecessary-lambda
      r = functional_ops.scan(
          lambda a, x: math_ops.multiply(a, x), elems, initializer=v)
      # pylint: enable=unnecessary-lambda
      r = gradients_impl.gradients(r, v)[0]
      self.assertAllEqual(873.0, self.evaluate(r))

  def testScanGradientWithPartStopGradient(self):
    a = variables.Variable(0.0, name="a")
    b = variables.Variable(0.0, name="b")
    elems = array_ops.zeros(5)
    l0, l1 = functional_ops.scan(
        lambda elem_, input_: (a, b), elems, initializer=(0., 0.))
    loss = l0 + array_ops.stop_gradient(l1)
    grad = gradients_impl.gradients(ys=[loss], xs=[a, b])
    with self.test_session(use_gpu=True) as sess:
      variables.global_variables_initializer().run()
      sess.run(grad)

  @test_util.run_in_graph_and_eager_modes
  def testFoldShape(self):
    with self.test_session():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])

      def fn(_, current_input):
        return current_input

      initializer = constant_op.constant([0, 0, 0])
      y = functional_ops.foldl(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  @test_util.run_in_graph_and_eager_modes
  def testMapShape(self):
    with self.test_session():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      y = functional_ops.map_fn(lambda e: e, x)
      self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  def testMapUnknownShape(self):
    x = array_ops.placeholder(dtypes.float32)
    y = functional_ops.map_fn(lambda e: e, x)
    self.assertIs(None, y.get_shape().dims)

  @test_util.run_in_graph_and_eager_modes
  def testMapEmptyScalar(self):
    with self.test_session():
      map_return = functional_ops.map_fn(lambda x: 1, constant_op.constant([]))
      self.assertAllEqual([0], map_return.get_shape().dims)
      self.assertAllEqual([0], self.evaluate(map_return).shape)

  # TODO(akshayka): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  def testMapEmptyTensor(self):
    with self.test_session():
      map_return = functional_ops.map_fn(lambda x: array_ops.zeros([3, 2]),
                                         constant_op.constant([]))
      self.assertAllEqual([0, 3, 2], map_return.get_shape().dims)
      self.assertAllEqual([0, 3, 2], self.evaluate(map_return).shape)

  @test_util.run_in_graph_and_eager_modes
  def testScanShape(self):
    with self.test_session():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])

      def fn(_, current_input):
        return current_input

      initializer = constant_op.constant([0, 0, 0])
      y = functional_ops.scan(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  # TODO(akshayka): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  def testScanEmptyTensor(self):
    with self.test_session():
      x = functional_ops.scan(
          lambda x, _: x, math_ops.range(0), initializer=array_ops.ones([2, 4]))
      self.assertAllEqual([0, 2, 4], x.get_shape())
      self.assertAllEqual(x.get_shape(), self.evaluate(x).shape)

  def testScanUnknownShape(self):
    x = array_ops.placeholder(dtypes.float32)
    initializer = array_ops.placeholder(dtypes.float32)

    def fn(_, current_input):
      return current_input

    y = functional_ops.scan(fn, x, initializer=initializer)
    self.assertIs(None, y.get_shape().dims)

  def testScanVaryingShape(self):
    with self.test_session() as sess:
      x = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 2])
      x_t = array_ops.transpose(x)
      # scan over dimension 0 (with shape None)
      result = functional_ops.scan(lambda a, x: a + x, x)
      # scanned over transposed dimension 0 (with shape 2)
      result_t = functional_ops.scan(lambda a, x: a + x, x_t, infer_shape=False)
      # ensure gradients can be calculated
      result_grad = gradients_impl.gradients(result, [x])[0]
      result_t_grad = gradients_impl.gradients(result_t, [x_t])[0]

      # smoke test to ensure they all evaluate
      sess.run([result, result_t, result_grad, result_t_grad],
               feed_dict={x: [[1.0, 2.0]]})

  def testRemoteFunction(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        1, 1, worker_config=worker_config)

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/job:ps/task:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/job:worker/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target="/job:worker/replica:0/task:0/cpu:1")

    with session.Session(worker[0].target) as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, [6])

  def testRemoteFunctionDirectSession(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/cpu:1")

    with self.test_session(config=worker_config) as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, [6])

  def testRemoteFunctionSameDeviceDirectSession(self):

    @function.Defun(dtypes.int32, dtypes.int32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.int32)
      b = variables.Variable(3, dtype=dtypes.int32)

    with ops.device("/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b], Tout=[dtypes.int32], f=_remote_fn, target="/cpu:0")

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, [6])

  def testRemoteFunctionCPUGPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/device:GPU:0")[0] + 3.0

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, 9.0)

  def testRemoteFunctionGPUCPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:localhost/replica:0/task:0/cpu:0")[0] + 3.0

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, 9.0)

  def testRemoteFunctionGPUCPUStrings(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    @function.Defun(dtypes.string)
    def _remote_fn(inp):
      return array_ops.identity(inp)

    a = array_ops.constant("a")

    with ops.device("/gpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a], Tout=[dtypes.string], f=_remote_fn, target="/cpu:0")

    with self.test_session() as sess:
      ret = sess.run(remote_op)
      self.assertAllEqual(ret, [b"a"])

  def testRemoteFunctionCrossProcess(self):
    workers, _ = test_util.create_local_cluster(2, 1)

    @function.Defun(dtypes.float32, dtypes.float32)
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    with ops.device("/job:ps/task:0"):
      a = variables.Variable(2, dtype=dtypes.float32)
      b = variables.Variable(3, dtype=dtypes.float32)

    with ops.device("/job:worker/replica:0/task:0/cpu:0"):
      remote_op = functional_ops.remote_call(
          args=[a, b],
          Tout=[dtypes.float32],
          f=_remote_fn,
          target="/job:worker/replica:0/task:1/cpu:0")[0] + 3.0

    with session.Session(workers[0].target) as sess:
      sess.run(variables.global_variables_initializer())
      mul = sess.run(remote_op)
      self.assertEqual(mul, 9)

  def testIf(self):

    @function.Defun(dtypes.float32)
    def Twice(x):
      return x * 2

    @function.Defun(dtypes.float32)
    def Thrice(x):
      return x * 3 + 1

    with self.test_session(use_gpu=False) as sess:

      x = array_ops.placeholder(dtypes.float32)
      ret = functional_ops.If(math_ops.greater(x, 0), [x], Twice, Thrice)[0]

      self.assertAllEqual(sess.run(ret, feed_dict={x: 9.}), 18.)
      self.assertAllEqual(sess.run(ret, feed_dict={x: -8.}), -23.)
      self.assertAllEqual(sess.run(ret, feed_dict={x: 0.}), 1.)

  def testWhile(self):

    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, unused_x):
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          return n - 1, x + n

        def Run(sess, n):
          return sess.run(functional_ops.While([n, 0.], Cond, Body))[1]

        with self.test_session(graph=g, use_gpu=use_gpu) as sess:
          self.assertAllEqual(Run(sess, 20.), 210.)
          self.assertAllEqual(Run(sess, 100.), 5050.)

  def testWhileError(self):
    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, unused_x):
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def CondReturnsTooManyArgs(n, x):
          return n > 0, x

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          return n - 1, x + n

        @function.Defun(*[dtypes.float32] * 2)
        def BodyReturnsTooManyArgs(n, x):
          return n - 1, x + n, x

        with self.test_session(graph=g, use_gpu=use_gpu):
          with self.assertRaisesRegexp(
              errors.InvalidArgumentError,
              "Expected a single scalar.*got 2 tensors."):
            functional_ops.While([5., 0.], CondReturnsTooManyArgs,
                                 Body)[0].eval()
          with self.assertRaisesRegexp(
              errors.InvalidArgumentError,
              "While loop body returned 3 arguments. Expected: 2"):
            functional_ops.While([5., 0.], Cond,
                                 BodyReturnsTooManyArgs)[0].eval()

  def testWhileInMultipleSubgraphs(self):

    for use_gpu in (True, False):
      with ops.Graph().as_default() as g:

        @function.Defun(*[dtypes.float32] * 2)
        def Cond(n, x):  # pylint: disable=unused-argument
          return n > 0

        @function.Defun(*[dtypes.float32] * 2)
        def Body(n, x):
          return n - 1, x + n

        with self.test_session(graph=g, use_gpu=use_gpu) as sess:
          n = array_ops.placeholder(dtypes.float32)
          _, result = functional_ops.While([n, 0.], Cond, Body)
          c = constant_op.constant(37.)

          self.assertAllEqual(210., sess.run(result, feed_dict={n: 20.}))
          self.assertAllEqual(5050., sess.run(result, feed_dict={n: 100.}))
          # Test that the result is the same when we run a different subgraph.
          self.assertAllEqual(5050.,
                              sess.run([result, c], feed_dict={n: 100.})[0])

  def _tfSum(self, use_gpu, rewrite_with_while):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g, use_gpu=use_gpu) as sess:

        @function.Defun(dtypes.int32, dtypes.float32)
        def Body(n, x):
          return x + math_ops.to_float(n)

        xs = [
            # 1 + 2  + ... + 20
            functional_ops.For(
                1, 21, 1, [0.], Body, rewrite_with_while=rewrite_with_while)[0],
            # 100 + 99 + ... + 1
            functional_ops.For(
                100, 0, -1, [0.], Body, rewrite_with_while=rewrite_with_while)
            [0],
        ]
        xvals = sess.run(xs)
      self.assertAllEqual(210, xvals[0])
      self.assertAllEqual(5050, xvals[1])

  def testFor(self):
    for use_gpu in (True, False):
      self._tfSum(use_gpu, False)

  def testForWithWhile(self):
    for use_gpu in (True, False):
      self._tfSum(use_gpu, True)

  def testForWithWhileNaming(self):
    g = ops.Graph()
    with g.as_default():

      @function.Defun(dtypes.int32, dtypes.float32, func_name="TestBody")
      def TestBody(n, x):
        return x + math_ops.to_float(n)

      _ = functional_ops.For(
          1, 21, 1, [0.], TestBody, rewrite_with_while=True)[0]

    names = []
    for func in g.as_graph_def().library.function:
      names.append(func.signature.name)
    self.assertTrue("TestBody" in names)
    self.assertTrue("TestBody_Cond" in names)
    self.assertTrue("TestBody_Body" in names)

  def testForCapturedInputs(self):
    v = variables.Variable(1.0)

    @function.Defun(dtypes.int32)
    def TestNullary(n):
      v + math_ops.to_float(n)  # pylint: disable=expression-not-assigned

    @function.Defun(dtypes.int32, dtypes.float32)
    def TestUnary(n, x):
      return x + math_ops.to_float(n) + v

    @function.Defun(dtypes.int32, dtypes.float32, dtypes.float32)
    def TestBinary(n, x, x2):
      return x + math_ops.to_float(n) + v, x2 + v

    for rewrite_with_while in (True, False):
      use_gpu = not rewrite_with_while
      with self.test_session(use_gpu=use_gpu) as sess:
        result_nullary = functional_ops.For(
            1, 10, 1, [], TestNullary,
            rewrite_with_while=rewrite_with_while)
        result_unary = functional_ops.For(
            1, 10, 1, [0.], TestUnary,
            rewrite_with_while=rewrite_with_while)
        result_binary = functional_ops.For(
            1, 10, 1, [0., 0.], TestBinary,
            rewrite_with_while=rewrite_with_while)
        sess.run(variables.global_variables_initializer())
        assert not result_nullary
        # The nullary variant doesn't return anything so we can't easily run it.
        # As a total hack, fetch the operation by name and run it.
        sess.run(ops.get_default_graph().get_operation_by_name(
            "While" if rewrite_with_while else "For"))
        assert len(result_unary) == 1
        self.assertEqual([54.0], sess.run(result_unary))
        assert len(result_binary) == 2
        self.assertEqual([54.0, 9.0], sess.run(result_binary))

  def _tfMLP(self, xval, wsval, bsval, rewrite_with_while):
    # On GPU, don't rewrite using a while loop.
    use_gpu = not rewrite_with_while
    with self.test_session(use_gpu=use_gpu):

      @function.Defun(dtypes.int32, *[dtypes.float64] * 3)
      def MLP(i, a, ws, bs):
        a = math_ops.tanh(math_ops.matmul(a, ws[i, :]) + bs[i, :])
        return a, ws, bs

      ret = functional_ops.For(
          0,
          wsval.shape[0],
          1, [xval, wsval, bsval],
          MLP,
          rewrite_with_while=rewrite_with_while)[0]

      return ret.eval()

  def _npMLP(self, xval, wsval, bsval):
    for i in range(wsval.shape[0]):
      xval = np.tanh(np.dot(xval, wsval[i, :]) + bsval[i, :])
    return xval

  def _testForMLP(self, rewrite_with_while):
    # We construct a 5-layer Multi-Layer Perceptron network here.
    # Each layer have the same number of hidden unites (3), and the
    # activation function is tanh().  We feed the input (xval) with
    # batch size 2.
    xval = np.random.normal(size=(2, 3))
    wsval = np.random.normal(size=(5, 3, 3))
    bsval = np.random.normal(size=(5, 3))
    np_ans = self._npMLP(xval, wsval, bsval)
    tf_for_ans = self._tfMLP(xval, wsval, bsval, rewrite_with_while)
    self.assertAllClose(np_ans, tf_for_ans)

  def testForMLP(self):
    self._testForMLP(False)

  def testForMLPWhile(self):
    self._testForMLP(True)

  def testForError(self):

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(i, v):
      return math_ops.to_float(i) + v

    @function.Defun(dtypes.int32, dtypes.float32)
    def ReturnsTooManyArgs(unused_i, v):
      return v, v

    with self.test_session(use_gpu=True):
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "must be a scalar"):
        functional_ops.For([0], 10, 1, [0.0], Foo)[0].eval()
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Invalid start/limit/delta"):
        functional_ops.For(0, 10, -1, [0.0], Foo)[0].eval()
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "For loop body returned 2 arguments. Expected: 1"):
        functional_ops.For(0, 10, 1, [0.0], ReturnsTooManyArgs)[0].eval()

  def testGradient(self):

    @function.Defun(dtypes.float32)
    def Poly(x):
      # y = 2x^3+3x^2+4x+8
      return 2 * x * x * x + 3 * x * x + 4 * x + 8

    @function.Defun(dtypes.float32)
    def Grad(x):
      # dy/dx = dy/dy * dy/dx = 1.0 * (6x^2+6x+4)
      return functional_ops.Gradient([x, 1.0], Poly)[0]

    with self.test_session(use_gpu=False) as sess:
      a = constant_op.constant(0.)
      avals = [Poly(a), Grad(a)]
      b = constant_op.constant(1.)
      bvals = [Poly(b), Grad(b)]
      self.assertAllEqual(sess.run(avals), [8., 4.])
      self.assertAllEqual(sess.run(bvals), [17., 16.])


# TODO(akshayka): Replace `function.Defun` with tf.contrib.eager.defun` in the
# below test cases.
class PartitionedCallTest(test.TestCase):

  def testBasicSingleDevice(self):

    @function.Defun(*[dtypes.float32] * 2)
    def Body(x, y):
      with ops.device("/cpu:0"):
        a = x + x
        b = y + y
        return a + b

    output, = self.evaluate(
        functional_ops.partitioned_call(
            args=[constant_op.constant(1.),
                  constant_op.constant(2.)], f=Body))
    self.assertEqual(output, 6.)

  def testBasicMultiDevice(self):
    config = config_pb2.ConfigProto(device_count={"CPU": 3})

    @function.Defun(*[dtypes.float32] * 2)
    def Body(x, y):
      # if x = 1, y = 2, ...
      with ops.device("/cpu:0"):
        # a:= 1 + 1 = 2
        a = x + x
      with ops.device("/cpu:1"):
        # b:= 2 + 2 = 4
        b = a + y
      with ops.device("/cpu:2"):
        # c:= 2 + 4 = 6
        c = a + b
      # a + b + c = 2 + 4 + 6 = 12
      return a + b + c

    with self.test_session(config=config):
      output, = functional_ops.partitioned_call(
          args=[constant_op.constant(1.),
                constant_op.constant(2.)], f=Body)
      self.assertEqual(output.eval(), 12.)

  def testBasicMultiDeviceGPU(self):
    if not test_util.is_gpu_available():
      return

    @function.Defun(*[dtypes.float32] * 2)
    def Body(x, y):
      with ops.device("/gpu:0"):
        a = x + x
        b = y + y
      with ops.device("/cpu:0"):
        c = a + b
        return c

    output, = self.evaluate(
        functional_ops.partitioned_call(
            args=[constant_op.constant(1.),
                  constant_op.constant(2.)], f=Body))
    self.assertEqual(output, 6.)

  def testBasicNoDeviceAnnotations(self):

    @function.Defun(*[dtypes.float32] * 2)
    def Body(x, y):
      a = x + x
      b = y + y
      return a + b

    output, = self.evaluate(
        functional_ops.partitioned_call(
            args=[constant_op.constant(1.),
                  constant_op.constant(2.)], f=Body))
    self.assertEqual(output, 6.)

  def testShardsRunOnRequestedDevices(self):
    config = config_pb2.ConfigProto(device_count={"CPU": 4})

    @function.Defun()
    def Body():
      # Serialize DT_RESOURCE handles as DT_STRINGs, which encode the device on
      # which the resource was created, so that we can verify that ops were
      # actually run on the requested devices.
      #
      # TODO(akshayka): Provide a cleaner, more idiomatic API for obtaining the
      # name of the device on which a resource lives / for determining the
      # device on which an op ran.
      with ops.device("/cpu:0"):
        s1 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      with ops.device("/cpu:1"):
        s2 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      with ops.device("/cpu:2"):
        s3 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      return s1, s2, s3

    with self.test_session(config=config, use_gpu=True) as sess:
      outputs = sess.run(functional_ops.partitioned_call(args=[], f=Body))
    self.assertIn(compat.as_bytes("CPU:0"), outputs[0])
    self.assertIn(compat.as_bytes("CPU:1"), outputs[1])
    self.assertIn(compat.as_bytes("CPU:2"), outputs[2])

  def testAssignAddResourceVariable(self):

    v = resource_variable_ops.ResourceVariable(1.0)

    @function.Defun()
    def AssignAdd():
      v.assign_add(1.0)

    op = functional_ops.partitioned_call(
        args=AssignAdd.captured_inputs, f=AssignAdd)
    _ = self.evaluate(variables.global_variables_initializer())
    _ = self.evaluate(op)
    value = self.evaluate(v.read_value())
    self.assertEqual(value, 2.0)

  def testFunctionWithResourcesOnDifferentDevices(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs available.")

    with ops.device("/cpu:0"):
      v_cpu_zero = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name="v_cpu_zero")

    with ops.device("/cpu:1"):
      v_cpu_one = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name="v_cpu_one")

    with ops.device("/gpu:0"):
      v_gpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name="v_gpu")

    def sum_gather():
      cpu_result = math_ops.reduce_sum(array_ops.gather(v_cpu_zero, [1, 2]))
      also_cpu_result = math_ops.reduce_sum(array_ops.gather(v_cpu_one, [1, 2]))
      gpu_result = math_ops.reduce_sum(array_ops.gather(v_gpu, [1, 2]))
      return cpu_result, also_cpu_result, gpu_result

    defined = function.Defun()(sum_gather)
    with self.test_session(
        config=config_pb2.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=True,
            device_count={"CPU": 2})) as sess:
      sess.run(variables.global_variables_initializer())
      expected = sess.run(sum_gather())
      result = sess.run(
          functional_ops.partitioned_call(
              args=defined.captured_inputs, f=defined))
      self.assertAllEqual(expected, result)


if __name__ == "__main__":
  test.main()

# pylint: enable=invalid-name
