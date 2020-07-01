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
"""Tests for tensorflow.ops.tf.gather_nd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GatherNdTest(test.TestCase):

  def _testSimpleDtype(self, dtype):
    with self.cached_session(use_gpu=True):
      params = constant_op.constant(np.array([8, 1, 2, 3, 7, 5], dtype=dtype))
      indices = constant_op.constant([[4], [4], [0]])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertAllEqual(np.array([7, 7, 8], dtype=dtype), gather_nd_val)
    self.assertEqual([3], gather_nd_t.get_shape())

  def testSimpleDtype(self):
    self._testSimpleDtype(np.float32)
    self._testSimpleDtype(np.float64)
    self._testSimpleDtype(np.int32)
    self._testSimpleDtype(np.int64)
    self._testSimpleDtype(np.complex64)
    self._testSimpleDtype(np.complex128)
    self._testSimpleDtype("|S")  # byte strings in python2 + 3

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testEmptyIndicesAndParamsOKButJustEmptyParamsFails(self):
    with self.session(use_gpu=True):
      params = np.ones((3, 3), dtype=np.float32)

      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      indices_empty = np.empty((0, 1), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0, 3], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0, 3), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params_empty, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_nonempty = np.zeros((1, 2), dtype=np.int32)
      gather_nd_break_t = array_ops.gather_nd(params_empty, indices_nonempty)
      with self.assertRaisesOpError(
          r"Requested more than 0 entries, but params is empty."):
        self.evaluate(gather_nd_break_t)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

  def testIndexScalar(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([4, 1])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)
      self.assertEqual([], gather_nd_t.get_shape())
      self.assertAllEqual(np.array(7), gather_nd_val)

  def testParamsRankLargerThanIndexIndexScalarSlices(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([4])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)
      self.assertEqual([2], gather_nd_t.get_shape())
      self.assertAllEqual(np.array([-7, 7]), gather_nd_val)

  def testParamsRankLargerThanIndexSlices(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([[4], [4], [0]])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([3, 2], gather_nd_t.get_shape())
    self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_val)

  def testHigherRankParamsLargerThanIndexSlices(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant([[4], [4], [0]])
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([3, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(params[[4, 4, 0]], gather_nd_val)

  def testEmptyIndicesLastRankMeansCopyEntireTensor(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant(
          [[], []], dtype=dtypes.int32)  # Size (2, 0)
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([2, 6, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(
        np.vstack((params[np.newaxis, :], params[np.newaxis, :])),
        gather_nd_val)

  def testHigherRankParamsAndIndicesLargerThanIndexSlices(self):
    with self.session(use_gpu=True):
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant([[[3], [2], [1]], [[4], [4], [0]]])
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([2, 3, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(params[[3, 2, 1, 4, 4, 0]].reshape(2, 3, 2, 2),
                        gather_nd_val)

  def testHigherRankParams(self):
    with self.session(use_gpu=True):
      shape = (10, 20, 5, 1, 17)
      params = np.random.rand(*shape)
      indices = np.vstack([np.random.randint(0, s, size=2000) for s in shape]).T
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected, gather_nd_val)
    self.assertEqual([2000], gather_nd_t.get_shape())

  def testHigherRankParamsAndIndices(self):
    with self.session(use_gpu=True):
      shape = (10, 20, 5, 1, 17)
      params = np.random.rand(*shape)
      indices = np.vstack([np.random.randint(0, s, size=2000) for s in shape]).T
      indices_reshaped = indices.reshape([10, 10, 20, 5])
      gather_nd_t = array_ops.gather_nd(params, indices_reshaped)
      gather_nd_val = self.evaluate(gather_nd_t)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected.reshape([10, 10, 20]), gather_nd_val)
    self.assertEqual([10, 10, 20], gather_nd_t.get_shape())

  def assertIndexedSlices(self, t):
    self.assertIsInstance(t, ops.IndexedSlices)

  @test_util.run_deprecated_v1
  def testUnknownIndices(self):
    params = constant_op.constant([[0, 1, 2]])
    indices = array_ops.placeholder(dtypes.int32)
    gather_nd_t = array_ops.gather_nd(params, indices)
    shape = gather_nd_t.get_shape()
    self.assertEqual(None, shape.ndims)
    self.assertEqual(None, tensor_shape.dimension_value(shape[0]))

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesCPU(self):
    with self.session(use_gpu=False):
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"):
        self.evaluate(gather_nd)

  def _disabledTestBadIndicesGPU(self):
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session(use_gpu=True):
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesWithSlicesCPU(self):
    with self.session(use_gpu=False):
      params = [[0, 1, 2]]
      indices = [[[0], [0], [1]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,2\] = \[1\] does not index into param shape \[1,3\]"):
        self.evaluate(gather_nd)

  def _disabledTestBadIndicesWithSlicesGPU(self):
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session(use_gpu=True):
      params = [[0, 1, 2]]
      indices = [[[0], [0], [1]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,2\] = \[1\] does not index into param shape \[1,3\]"):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  def testGradientsRank2Elements(self):
    indices = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.int32)
    inputs = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant([1, 2], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array([[1, 0], [0, 2]], dtype=np.float64)
    with self.session(use_gpu=True):
      assert np.array_equal(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank2Slices(self):
    indices = constant_op.constant([[1], [0]], dtype=dtypes.int32)
    inputs = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array([[3, 4], [1, 2]], dtype=np.float64)
    with self.session(use_gpu=True):
      self.assertIndexedSlices(grads)
      self.assertAllEqual(expected_grads, ops.convert_to_tensor(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank3Elements(self):
    indices = constant_op.constant(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[5, 6], [1, 2]], [[3, 4], [7, 8]]], dtype=np.float64)
    with self.session(use_gpu=True):
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank7Elements(self):
    # Shape [1,1,2,1,1,2,2]
    indices = constant_op.constant(
        [[[
            [[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]
        ]]],
        dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[[
            [[[[1, 3], [5, 7]]]],
            [[[[2, 4], [6, 8]]]]
        ]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[
            [[[[1, 2], [3, 4]]]],
            [[[[5, 6], [7, 8]]]]
        ]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[
            [[[[5, 6], [1, 2]]]],
            [[[[3, 4], [7, 8]]]]
        ]]], dtype=np.float64)
    with self.session(use_gpu=True):
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsInt64Indices(self):
    indices = constant_op.constant(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=dtypes.int64)
    inputs = constant_op.constant(
        [[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[5, 6], [1, 2]], [[3, 4], [7, 8]]], dtype=np.float64)
    with self.session(use_gpu=True):
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank2SlicesWithEmptySpace(self):
    indices = constant_op.constant([[2], [0], [5]], dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)
    grad_vals = constant_op.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3]],
        dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3]],
        dtype=np.float64)
    with self.session(use_gpu=True):
      self.assertIndexedSlices(grads)
      self.assertAllEqual(expected_grads, ops.convert_to_tensor(grads))

  @test_util.run_v1_only("RefVariable is not supported in v2")
  def testGatherNdRefVariable(self):
    with self.cached_session():
      v = variables.RefVariable(constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather_nd(v, [[0, 1], [2, 0]])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("GatherNd", gather.op.name)
      self.assertAllEqual([2, 5], gather)

  @test_util.run_in_graph_and_eager_modes
  def testGatherNdResourceVariable(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather_nd(v, [[0, 1], [2, 0]])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("ResourceGatherNd", gather.op.inputs[0].op.type)
      self.assertAllEqual([2, 5], gather)


class GatherNdOpBenchmark(test.Benchmark):

  def benchmark_gather_nd_op(self):
    shape = (100, 47, 18, 170, 13)
    np.random.seed(127)
    params = np.random.rand(*shape)
    indices = np.vstack([np.random.randint(0, s, size=10000) for s in shape]).T

    with session.Session():
      t_params = variables.Variable(params)
      t_indices = variables.Variable(indices)
      gather_op = array_ops.gather_nd(t_params, t_indices)
      variables.global_variables_initializer().run()
      for _ in range(10):
        self.evaluate(gather_op)
      t1 = time.time()
      for _ in range(1000):
        self.evaluate(gather_op)
      t2 = time.time()
      self.report_benchmark(iters=1000, wall_time=(t2 - t1) / 1000.0)


if __name__ == "__main__":
  test.main()
