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
"""Tests for tensorflow.ops.tf.gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test

_TEST_TYPES = (dtypes.int64, dtypes.float32,
               dtypes.complex64, dtypes.complex128)


class GatherTest(test.TestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    with self.test_session(use_gpu=True):
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in _TEST_TYPES:
        for indices in 4, [1, 2, 2, 4, 5]:
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices_tf = constant_op.constant(indices)
          gather_t = array_ops.gather(params, indices_tf)
          gather_val = gather_t.eval()
          np_val = params_np[indices]
          self.assertAllEqual(np_val, gather_val)
          self.assertEqual(np_val.shape, gather_t.get_shape())

  def testScalar2D(self):
    with self.test_session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices = constant_op.constant(2)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = gather_t.eval()
          self.assertAllEqual(np.take(params_np, 2, axis=axis), gather_val)
          expected_shape = data.shape[:axis] + data.shape[axis + 1:]
          self.assertEqual(expected_shape, gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.test_session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = gather_t.eval()
          self.assertAllEqual(np.take(params_np, [0, 1, 0, 2], axis=axis),
                              gather_val)
          expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1:]
          self.assertEqual(expected_shape, gather_t.get_shape())

  def testHigherRank(self):
    # We check that scalar and empty indices shapes work as well
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in _TEST_TYPES:
        for axis in range(len(shape)):
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.test_session(use_gpu=True) as sess:
            tf_params = constant_op.constant(params)
            tf_indices = constant_op.constant(indices)
            # Check that both positive and negative indices for axis work.
            tf_axis = constant_op.constant(axis)
            tf_negative_axis = constant_op.constant(-len(shape) + axis)
            gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
            gather_negative_axis = array_ops.gather(
                tf_params, tf_indices, axis=tf_negative_axis)
            gather_value, gather_negative_axis_value = sess.run(
                [gather, gather_negative_axis])
            gather_np = np.take(params, indices, axis)
            self.assertAllEqual(gather_np, gather_value)
            self.assertAllEqual(gather_np, gather_negative_axis_value)
            expected_shape = (params.shape[:axis] + indices.shape +
                              params.shape[axis + 1:])
            self.assertEqual(expected_shape, gather.shape)
            self.assertEqual(expected_shape, gather_negative_axis.shape)

            # Test gradients
            gather_grad = np.random.randn(
                *gather.get_shape().as_list()).astype(dtype.as_numpy_dtype)
            if dtype.is_complex:
              gather_grad -= 1j * gather_grad
            params_grad, indices_grad, axis_grad = gradients_impl.gradients(
                gather, [tf_params, tf_indices, tf_axis], gather_grad)
            self.assertEqual(indices_grad, None)
            self.assertEqual(axis_grad, None)
            if dtype.is_integer:
              self.assertEqual(params_grad, None)
              continue
            # For axis 0, we are able to create an efficient IndexedSlices for
            # the gradient.
            if axis == 0:
              self.assertEqual(type(params_grad), ops.IndexedSlices)
              params_grad = ops.convert_to_tensor(params_grad)
            correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
            outer_dims = axis
            inner_dims = len(shape) - axis - 1
            gather_grad = gather_grad.reshape(
                shape[:axis] + (indices.size,) + shape[axis + 1:])
            for source_index, dest_index in enumerate(indices.flat):
              dest_slice = ((slice(None),) * outer_dims + (dest_index,) +
                            (slice(None),) * inner_dims)
              source_slice = ((slice(None),) * outer_dims + (source_index,) +
                              (slice(None),) * inner_dims)
              correct_params_grad[dest_slice] += gather_grad[source_slice]
            self.assertAllClose(correct_params_grad, params_grad.eval(),
                                atol=2e-6, rtol=2e-6)

  def testString(self):
    params = np.array([[b"asdf", b"zxcv"], [b"qwer", b"uiop"]])
    with self.test_session():
      self.assertAllEqual([b"qwer", b"uiop"],
                          array_ops.gather(params, 1, axis=0).eval())
      self.assertAllEqual([b"asdf", b"qwer"],
                          array_ops.gather(params, 0, axis=1).eval())

  def testUInt32AndUInt64(self):
    for unsigned_type in (dtypes.uint32, dtypes.uint64):
      params = self._buildParams(
          np.array([[1, 2, 3], [7, 8, 9]]), unsigned_type)
      with self.test_session():
        self.assertAllEqual([7, 8, 9],
                            array_ops.gather(params, 1, axis=0).eval())
        self.assertAllEqual([1, 7], array_ops.gather(params, 0, axis=1).eval())

  def testUnknownIndices(self):
    params = constant_op.constant([[0, 1, 2]])
    indices = array_ops.placeholder(dtypes.int32)
    gather_t = array_ops.gather(params, indices)
    self.assertEqual(None, gather_t.get_shape())

  def testUnknownAxis(self):
    params = constant_op.constant([[0, 1, 2]])
    indices = constant_op.constant([[0, 0], [0, 0]])
    axis = array_ops.placeholder(dtypes.int32)
    gather_t = array_ops.gather(params, indices, axis=axis)
    # Rank 2 params with rank 2 indices results in a rank 3 shape.
    self.assertEqual([None, None, None], gather_t.shape.as_list())

    # If indices is also unknown the result rank is unknown.
    indices = array_ops.placeholder(dtypes.int32)
    gather_t = array_ops.gather(params, indices, axis=axis)
    self.assertEqual(None, gather_t.shape)

  def testBadIndicesCPU(self):
    with self.test_session(use_gpu=False):
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        array_ops.gather(params, [[7]], axis=0).eval()
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        array_ops.gather(params, [[7]], axis=1).eval()

  def _disabledTestBadIndicesGPU(self):
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.test_session(use_gpu=True):
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        array_ops.gather(params, [[7]], axis=0).eval()
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        array_ops.gather(params, [[7]], axis=1).eval()

  def testBadAxis(self):
    with self.test_session(use_gpu=True):
      params = [0, 1, 2]
      params_ph = array_ops.placeholder(dtypes.int32)
      indices = 0
      for bad_axis in (1, 2, -2):
        # Shape inference can validate axis for known params rank.
        with self.assertRaisesWithPredicateMatch(
            ValueError, "Shape must be at least rank . but is rank 1"):
          array_ops.gather(params, indices, axis=bad_axis)
        # If params rank is unknown, an op error occurs.
        with self.assertRaisesOpError(
            r"Expected axis in the range \[-1, 1\), but got %s" % bad_axis):
          array_ops.gather(params_ph, indices, axis=bad_axis).eval(
              feed_dict={params_ph: params})

  def testEmptySlices(self):
    with self.test_session(use_gpu=True):
      for dtype in _TEST_TYPES:
        for itype in np.int32, np.int64:
          # Leading axis gather.
          params = np.zeros((7, 0, 0), dtype=dtype.as_numpy_dtype)
          indices = np.array([3, 4], dtype=itype)
          gather = array_ops.gather(params, indices, axis=0)
          self.assertAllEqual(gather.eval(), np.zeros((2, 0, 0)))

          # Middle axis gather.
          params = np.zeros((0, 7, 0), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=1)
          self.assertAllEqual(gather.eval(), np.zeros((0, 2, 0)))

          # Trailing axis gather.
          params = np.zeros((0, 0, 7), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=2)
          self.assertAllEqual(gather.eval(), np.zeros((0, 0, 2)))


if __name__ == "__main__":
  test.main()
