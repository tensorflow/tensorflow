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

_TEST_TYPES = (dtypes.float32, dtypes.complex64, dtypes.complex128)


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
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices = constant_op.constant(4)
        gather_t = array_ops.gather(params, indices)
        gather_val = gather_t.eval()
        self.assertAllEqual(params_np[4], gather_val)
        self.assertEqual([], gather_t.get_shape())

  def testScalar2D(self):
    with self.test_session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices = constant_op.constant(2)
        gather_t = array_ops.gather(params, indices)
        gather_val = gather_t.eval()
        self.assertAllEqual(params_np[2], gather_val)
        self.assertEqual([3], gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.test_session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices = constant_op.constant([0, 4, 0, 2])
        gather_t = array_ops.gather(params, indices)
        gather_val = gather_t.eval()
        self.assertAllEqual(params_np[[0, 4, 0, 2]], gather_val)
        self.assertEqual([4, 3], gather_t.get_shape())

  def testHigherRank(self):
    np.random.seed(1)
    # We check that scalar and empty shapes work as well
    for shape in (7, 0), (4, 3, 2):
      for indices_shape in (), (0,), (3, 0), (3, 5):
        for dtype in _TEST_TYPES:
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[0], size=indices_shape)
          with self.test_session(use_gpu=True):
            tf_params = constant_op.constant(params)
            tf_indices = constant_op.constant(indices)
            gather = array_ops.gather(tf_params, tf_indices)
            self.assertAllEqual(params[indices], gather.eval())
            self.assertEqual(indices.shape + params.shape[1:],
                             gather.get_shape())
            # Test gradients
            gather_grad = np.random.randn(*gather.get_shape().as_list()).astype(
                dtype.as_numpy_dtype)
            if dtype.is_complex:
              gather_grad -= 1j * gather_grad
            params_grad, indices_grad = gradients_impl.gradients(
                gather, [tf_params, tf_indices], gather_grad)
            self.assertEqual(indices_grad, None)
            self.assertEqual(type(params_grad), ops.IndexedSlices)
            params_grad = ops.convert_to_tensor(params_grad)
            correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
            for i, g in zip(indices.flat,
                            gather_grad.reshape((indices.size,) + shape[1:])):
              correct_params_grad[i] += g
            self.assertAllClose(correct_params_grad, params_grad.eval())

  def testUnknownIndices(self):
    params = constant_op.constant([[0, 1, 2]])
    indices = array_ops.placeholder(dtypes.int32)
    gather_t = array_ops.gather(params, indices)
    self.assertEqual(None, gather_t.get_shape())

  def testBadIndices(self):
    with self.test_session(use_gpu=True):
      params = [0, 1, 2]
      indices = [[7]]
      gather = array_ops.gather(params, indices)
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        gather.eval()

  def testEmptySlices(self):
    with self.test_session(use_gpu=True):
      for dtype in _TEST_TYPES:
        for itype in np.int32, np.int64:
          params = np.zeros((7, 0), dtype=dtype.as_numpy_dtype)
          indices = np.array([3, 4], dtype=itype)
          gather = array_ops.gather(params, indices)
          self.assertAllEqual(gather.eval(), np.zeros((2, 0)))


if __name__ == "__main__":
  test.main()
