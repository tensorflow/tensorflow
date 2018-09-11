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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


def _indexedslice(x, noshape=False):
  x = np.array(x)
  dense_shape = x.shape
  ndim = len(dense_shape)
  indices = np.where(np.sum(x, tuple(range(1, ndim))))[0]
  values = x[indices]
  if noshape:
    dense_shape = None
  return ops.IndexedSlices(
      indices=indices.tolist(), values=values, dense_shape=dense_shape)


class IndexedSlicesConditionalAccumulatorTest(test.TestCase):

  def _assertEqual_indexedslices(self, expected_tensor, result):
    self.assertAllEqual(expected_tensor.indices, result.indices)
    self.assertAllEqual(expected_tensor.values, result.values)
    if (result.dense_shape is not None and
        expected_tensor.dense_shape is not None):
      self.assertAllEqual(expected_tensor.dense_shape, result.dense_shape)

  def _assertEqual_nparray(self, expected_array, result, sess):
    expected_tensor = _indexedslice(expected_array)
    self._assertEqual_indexedslices(expected_tensor, result)

  def testConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q")
    self.assertTrue(isinstance(q.accumulator_ref, ops.Tensor))
    self.assertProtoEquals(
        """
      name:'Q' op:'SparseConditionalAccumulator'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'shape' value { shape { unknown_rank: true} } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      attr { key: 'reduction_type' value {s: 'MEAN'} }
      """, q.accumulator_ref.op.node_def)

  def testConstructorWithInvalidArg(self):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        data_flow_ops.SparseConditionalAccumulator(
            dtypes_lib.float32, name="Q", reduction_type="Invalid")

  def testConstructorWithShape(self):
    with ops.Graph().as_default():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1, 5, 2, 8]))
    self.assertTrue(isinstance(q.accumulator_ref, ops.Tensor))
    self.assertProtoEquals(
        """
      name:'Q' op:'SparseConditionalAccumulator'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'shape' value { shape { dim {size: 1 }
                                          dim {size: 5 }
                                          dim {size: 2 }
                                          dim {size: 8 }
      } } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      attr { key: 'reduction_type' value {s: 'MEAN'} }
      """, q.accumulator_ref.op.node_def)

  def testAccumulatorSizeEmpty(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q")
      self.assertEqual(q.num_accumulated().eval(), 0)

  def testAccumulatorSetGlobalStep(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([1]))
      set_global_step_op = q.set_global_step(1)
      set_global_step_op.run()

  def testAccumulatorApplyGradFloat32(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))
      accum_op = q.apply_indexed_slices_grad(
          ops.IndexedSlices(
              indices=[0, 2],
              values=np.array([[0, 0, 1], [3, 0, 4]]).astype(np.float32)))
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 1)

  def testDtypes(self):
    with self.test_session() as sess:
      dtypes = [dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64]

      for i in range(len(dtypes)):
        dtype = dtypes[i]
        q = data_flow_ops.SparseConditionalAccumulator(
            dtype, shape=tensor_shape.TensorShape([3, 3, 3]))

        elems = np.arange(2)
        sum_elems = np.zeros([3, 3, 3]).astype(dtype.as_numpy_dtype)
        for e in elems:
          mat_to_add = np.zeros([3, 3, 3]).astype(dtype.as_numpy_dtype)
          mat_to_add[i, i, i] = e + 1
          sum_elems += mat_to_add
          t = _indexedslice(mat_to_add)
          q.apply_indexed_slices_grad(t).run()

        result = sess.run(q.take_indexed_slices_grad(1))

        self._assertEqual_nparray(sum_elems / len(elems), result, sess)

  def testAccumulatorMultipleAccumulators(self):
    with self.test_session() as sess:
      q_f32_0 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f32_1 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f16_0 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      q_f16_1 = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float16, name="Q", shape=tensor_shape.TensorShape([2, 2]))

      accums = [q_f16_0, q_f16_1, q_f32_0, q_f32_1]

      elems = [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0],
                                                                      [0, 1]]]

      expected_tensors = []

      for i in range(len(accums)):
        tensor_to_add = np.array(elems[i]).astype(accums[i]
                                                  .dtype.as_numpy_dtype)
        expected_tensor = _indexedslice(tensor_to_add)
        expected_tensors.append(expected_tensor)
        st = _indexedslice(tensor_to_add)
        accums[i].apply_indexed_slices_grad(st).run()

      for i in range(len(accums)):
        result = sess.run(accums[i].take_indexed_slices_grad(1))
        self._assertEqual_indexedslices(expected_tensors[i], result)

  def testAccumulatorTakeGradMean(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=())

      grad_indexed_slices = ops.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices)
      accum_op.run()
      accum_op = q.apply_grad([0, 2],
                              np.array([[0, 1], [3, 0]]).astype(np.float32),
                              [3, 2])
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = sess.run(takeg_t)
      self.assertAllEqual([0, 1, 2], val.indices)
      self.assertAllEqual([[0.5, 0.5], [0, 2], [3, 0]], val.values)
      self.assertAllEqual([-1, 2], val.dense_shape)

  def testAccumulatorTakeGradSum(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(), reduction_type="SUM")

      grad_indexed_slices = ops.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices)
      accum_op.run()
      accum_op = q.apply_grad([0, 2],
                              np.array([[0, 1], [3, 0]]).astype(np.float32),
                              [3, 2])
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = sess.run(takeg_t)
      self.assertAllEqual([0, 1, 2], val.indices)
      self.assertAllEqual([[1, 1], [0, 2], [3, 0]], val.values)
      self.assertAllEqual([-1, 2], val.dense_shape)

  def testAccumulatorTakeGradInvalidReductionType(self):
    with self.assertRaises(ValueError):
      data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(), reduction_type="Invalid")

  def testAccumulatorRepeatedTakeGrad(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=())

      grad_indexed_slices = ops.IndexedSlices(
          indices=[0, 1], values=np.array([[1, 0], [0, 2]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices, local_step=0)
      accum_op.run()
      accum_op = q.apply_grad(
          [0, 2],
          np.array([[0, 1], [3, 0]]).astype(np.float32), [3, 2],
          local_step=0)
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = sess.run(takeg_t)
      self.assertAllEqual(val.indices, [0, 1, 2])
      self.assertAllEqual(val.values, [[0.5, 0.5], [0, 2], [3, 0]])
      self.assertAllEqual(val.dense_shape, [-1, 2])

      grad_indexed_slices = ops.IndexedSlices(
          indices=[0, 1],
          values=np.array([[10, 0], [0, 20]]).astype(np.float32))
      accum_op = q.apply_indexed_slices_grad(grad_indexed_slices, local_step=1)
      accum_op.run()
      accum_op = q.apply_grad(
          [0, 2],
          np.array([[0, 10], [30, 0]]).astype(np.float32), [3, 2],
          local_step=1)
      accum_op.run()

      takeg_t = q.take_indexed_slices_grad(1)
      val = sess.run(takeg_t)
      self.assertAllEqual(val.indices, [0, 1, 2])
      self.assertAllEqual(val.values, [[5, 5], [0, 20], [30, 0]])
      self.assertAllEqual(val.dense_shape, [-1, 2])

  def testParallelApplyGradMean(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[x, 0], [0, x]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(1)

      def apply_indexed_slices_grad(accum_op):
        sess.run(accum_op)

      threads = [
          self.checkedThread(
              target=apply_indexed_slices_grad, args=(o,)) for o in accum_ops
      ]

      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      val = sess.run(takeg_t)

      expected_val = sum(elems) / len(elems)
      self._assertEqual_nparray(
          np.array([[expected_val, 0], [0, expected_val]]).astype(np.float32),
          val, sess)

  def testParallelApplyGradSum(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([2, 2]),
          reduction_type="SUM")
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[x, 0], [0, x]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(1)

      def apply_indexed_slices_grad(accum_op):
        sess.run(accum_op)

      threads = [
          self.checkedThread(target=apply_indexed_slices_grad, args=(o,))
          for o in accum_ops
      ]

      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      val = sess.run(takeg_t)

      expected_val = 550.0
      self._assertEqual_nparray(
          np.array([[expected_val, 0], [0, expected_val]]).astype(np.float32),
          val, sess)

  def testParallelTakeGrad(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))
      elems = [e + 1 for e in range(10)]
      accum_ops = []
      for e in elems:
        v = _indexedslice(np.array([[0, 0], [e, 0]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(v, local_step=e - 1))
      takeg_t = q.take_indexed_slices_grad(1)

      results = []

      def apply_indexed_slices_grad():
        for accum_op in accum_ops:
          time.sleep(1.0)
          sess.run(accum_op)

      apply_indexed_slices_grad_thread = self.checkedThread(
          target=apply_indexed_slices_grad)

      def take_grad():
        t = sess.run(takeg_t)
        results.append(t)

      threads = [self.checkedThread(target=take_grad) for _ in range(10)]

      for thread in threads:
        thread.start()
      apply_indexed_slices_grad_thread.start()

      for thread in threads:
        thread.join()
      apply_indexed_slices_grad_thread.join()

      for i in range(len(accum_ops)):
        self._assertEqual_nparray(
            np.array([[0, 0], [elems[i], 0]]), results[i], sess)

  def testAccumulatorApplyAndBlockingTake(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([2, 2]))

      elems = [10.0, 20.0, 30.0]
      elems_ave = sum(elems) / len(elems)
      accum_ops = []
      for x in elems:
        x = _indexedslice(np.array([[0, x], [0, 0]]).astype(np.float32))
        accum_ops.append(q.apply_indexed_slices_grad(x, local_step=0))
      takeg_t = q.take_indexed_slices_grad(3)

      results = []

      def apply_indexed_slices_grad():
        for accum_op in accum_ops:
          sess.run(accum_op)

      def take_grad():
        results.append(sess.run(takeg_t))

      accum_thread = self.checkedThread(target=apply_indexed_slices_grad)
      takeg_thread = self.checkedThread(target=take_grad)
      accum_thread.start()
      takeg_thread.start()
      accum_thread.join()
      takeg_thread.join()

      self._assertEqual_nparray([[0, elems_ave], [0, 0]], results[0], sess)

  def _blocking_takeg(self, sess, takeg_op):
    with self.assertRaisesOpError("was cancelled"):
      sess.run(takeg_op)

  def testAccumulatorCancel(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32,
          name="Q",
          shape=tensor_shape.TensorShape([1, 2, 3]))
      takeg_t = q.take_indexed_slices_grad(1)

      takeg_thread = self.checkedThread(
          self._blocking_takeg, args=(sess, takeg_t))

      takeg_thread.start()

      time.sleep(1.0)

      sess.close()  # Will cancel blocked operation

      takeg_thread.join()

  def testNonVectorIndices(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Input indices should be vector but received shape:"):
        q.apply_grad(
            grad_indices=[[0, 1], [1, 0]],
            grad_values=np.array([1, 2]).astype(np.float32)).run()

  def testZeroDimensionValues(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Values cannot be 0-dimensional."):
        q.apply_grad(
            grad_indices=[0], grad_values=np.array(1).astype(np.float32)).run()

  def testWrongNonEmptyInputValues(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   " non-empty input values, got "):
        q.apply_grad(
            grad_indices=[0, 1],
            grad_values=np.array([[0, 1, 1]]).astype(np.float32)).run()

  def testDynamicNonVectorIndices(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      x_indices = array_ops.placeholder(dtypes_lib.int64)
      x_values = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(grad_indices=x_indices, grad_values=x_values)

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Input indices should be vector but received shape:"):
        sess.run(accum_op,
                 feed_dict={
                     x_indices: [[0, 1], [1, 0]],
                     x_values: np.array([1, 2]).astype(np.float32)
                 })

  def testDynamicWrongNonEmptyInputValues(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))

      x_indices = array_ops.placeholder(dtypes_lib.int64)
      x_values = array_ops.placeholder(dtypes_lib.float32)

      accum_op = q.apply_grad(grad_indices=x_indices, grad_values=x_values)

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   " non-empty input values, got "):
        sess.run(accum_op,
                 feed_dict={
                     x_indices: [0, 1],
                     x_values: np.array([[0, 1, 1]]).astype(np.float32)
                 })

  def testEmptyShapeApply(self):
    with self.test_session():
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([]))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Input indices should be vector"):
        q.apply_grad(grad_indices=0, grad_values=[1.0], grad_shape=[]).run()

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Input indices should be vector"):
        q.apply_grad(grad_indices=0, grad_values=[1.0]).run()

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Values cannot be 0-dimensional."):
        q.apply_grad(grad_indices=[0], grad_values=1.0, grad_shape=[]).run()

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Values cannot be 0-dimensional."):
        q.apply_grad(grad_indices=[0], grad_values=1.0).run()

      # The right way to apply a scalar
      q.apply_grad(grad_indices=[0], grad_values=[1.0], grad_shape=[]).run()
      q.apply_grad(grad_indices=[0], grad_values=[1.0]).run()

  def testValidateShape(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[2, 2, None])

      # Provided shape has wrong rank
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected shape rank at least 3, got 2"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[1, 2]]).astype(np.float32),
            grad_shape=[2, 2]).run()

      # Provided shape has wrong dim
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected shape dim 1 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32),
            grad_shape=[2, 3, 2]).run()

      # Indices exceeded accumulator's shape's limits
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: index of slice 0 exceeded limits of shape;"
          " index is 3 exceeded 2"):
        q.apply_grad(
            grad_indices=[3],
            grad_values=np.array([[[1, 2], [3, 4]]]).astype(np.float32)).run()

      # Values' rank does not match shape
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values rank at least 3, got 2"):
        q.apply_grad(
            grad_indices=[0, 1],
            grad_values=np.array([[1, 2], [3, 4]]).astype(np.float32)).run()

      # Values' dim does not match shape
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 1 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)).run()

      # First successful gradient creates additional constraints
      # Shape will be additionally be constrained to [None,2,2,2] hereafter.
      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)).run()

      # Values' rank does not match accumulated gradient
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values rank 4, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array([[[1, 2], [3, 4]]]).astype(np.float32)).run()

      # Values' dim does not match accumulated gradient
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 3 to be 2, got 3"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                    np.float32)).run()

      # After take grad, constraints on accumulated gradient are removed
      sess.run(q.take_grad(1))

      # First successful gradient imposes new constraints.
      # Hereafter, shape will additionally constrained to [None,2,2,3]
      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                  np.float32),
          local_step=1).run()

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Shape mismatch: expected values dim 3 to be 3, got 2"):
        q.apply_grad(
            grad_indices=[0],
            grad_values=np.array(
                [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32),
            local_step=1).run()

  def testReturnShape(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[2, None])

      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)).run()

      val = sess.run(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.dense_shape, [2, 2, 2, 2])

      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=[None, 2])

      q.apply_grad(
          grad_indices=[0],
          grad_values=np.array(
              [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(
                  np.float32)).run()

      val = sess.run(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.dense_shape, [-1, 2, 2, 3])

  def testApplyGradtInt32IndicesAndShape(self):
    with self.test_session() as sess:
      q = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=tensor_shape.TensorShape([3, 3]))
      accum_op = q.apply_grad(
          grad_indices=constant_op.constant(
              [0, 2], dtype=dtypes_lib.int32),
          grad_values=constant_op.constant(
              [[0, 0, 1], [3, 0, 4]], dtype=dtypes_lib.float32),
          grad_shape=constant_op.constant(
              [3, 3], dtype=dtypes_lib.int32))
      accum_op.run()
      accum_op = q.apply_indexed_slices_grad(
          ops.IndexedSlices(
              indices=constant_op.constant(
                  [0, 2], dtype=dtypes_lib.int32),
              values=constant_op.constant(
                  [[0, 0, 1], [3, 0, 4]], dtype=dtypes_lib.float32),
              dense_shape=constant_op.constant(
                  [3, 3], dtype=dtypes_lib.int32)))
      accum_op.run()
      self.assertEqual(q.num_accumulated().eval(), 2)

      val = sess.run(q.take_indexed_slices_grad(1))
      self.assertAllEqual(val.indices, [0, 2])
      self.assertAllEqual(val.values, [[0, 0, 1], [3, 0, 4]])
      self.assertAllEqual(val.dense_shape, [3, 3])


if __name__ == "__main__":
  test.main()
