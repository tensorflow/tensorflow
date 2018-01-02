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
"""Tests for tensorflow.kernels.bcast_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


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

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testMap_SimpleNotTensor(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(
          lambda x: math_ops.multiply(math_ops.add(x, 3), 2), nums)
      self.assertAllEqual(
          np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testMap_MultiOutputMismatchedDtype(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      with self.assertRaisesRegexp(
          TypeError, r"two structures don't have the same sequence type."):
        # lambda emits tuple, but dtype is a list
        functional_ops.map_fn(
            lambda x: ((x + 3) * 2, -(x + 3) * 2),
            nums,
            dtype=[dtypes.int64, dtypes.int64])

  @test_util.run_in_graph_and_eager_modes()
  def testMap_MultiInputSingleOutput(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = functional_ops.map_fn(
          lambda x: x[0] * x[1][0] + x[1][1], (nums, (nums, -nums)),
          dtype=dtypes.int64)
      self.assertEqual((6,), r.get_shape())
      received = self.evaluate(r)
      self.assertAllEqual(nums * nums + (-nums), received)

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testScan_SingleInputMultiOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = (np.array(1.0), np.array(-1.0))
      r = functional_ops.scan(lambda a, x: (a[0] * x, -a[1] * x), elems,
                              initializer)
      r_value = self.evaluate(r)

      self.assertAllEqual([1.0, 2.0, 6.0, 24.0, 120.0, 720.0], r_value[0])
      self.assertAllEqual([1.0, -2.0, 6.0, -24.0, 120.0, -720.0], r_value[1])

  @test_util.run_in_graph_and_eager_modes()
  def testScan_MultiInputSingleOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      r = functional_ops.scan(lambda a, x: a * (x[0] + x[1]),
                              (elems + 1, -elems), initializer)
      self.assertAllEqual([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes()
  def testScan_MultiInputSameTypeOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      r = functional_ops.scan(lambda a, x: (a[0] + x[0], a[1] + x[1]),
                              (elems, -elems))
      r_value = self.evaluate(r)
      self.assertAllEqual(np.cumsum(elems), r_value[0])
      self.assertAllEqual(np.cumsum(-elems), r_value[1])

  @test_util.run_in_graph_and_eager_modes()
  def testScan_MultiOutputMismatchedInitializer(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      with self.assertRaisesRegexp(
          ValueError, "two structures don't have the same number of elements"):
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

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testFoldShape(self):
    with self.test_session():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])

      def fn(_, current_input):
        return current_input

      initializer = constant_op.constant([0, 0, 0])
      y = functional_ops.foldl(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  @test_util.run_in_graph_and_eager_modes()
  def testMapShape(self):
    with self.test_session():
      x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      y = functional_ops.map_fn(lambda e: e, x)
      self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  def testMapUnknownShape(self):
    x = array_ops.placeholder(dtypes.float32)
    y = functional_ops.map_fn(lambda e: e, x)
    self.assertIs(None, y.get_shape().dims)

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
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


if __name__ == "__main__":
  test.main()
