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
"""Functional tests for SparseTensorDenseMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import test


def _maybe_complex(x):
  if x.dtype.kind == "c":  # complex
    return (x + 1j * x) / 2
  return x


class SparseTensorDenseMatMulTestBase(test.TestCase):

  def _testMatmul(self,
                  x,
                  y,
                  adjoint_a=False,
                  adjoint_b=False,
                  indices_dtype=np.int64):
    x_mat = np.matrix(x)
    if adjoint_a:
      x_mat = x_mat.H
    y_mat = np.matrix(y)
    if adjoint_b:
      y_mat = y_mat.H

    np_ans = x_mat * y_mat

    x_indices = np.vstack(np.where(x)).astype(indices_dtype).T
    x_values = x[np.where(x)]
    x_shape = x.shape

    with self.cached_session(use_gpu=True):
      determinism_required = False
      if os.getenv('TF_DETERMINISTIC_OPS') in ('1', 'True'):
        determinism_required = True

      sp_x_value = sparse_tensor.SparseTensorValue(
          indices=x_indices, values=x_values, dense_shape=x_shape)
      gpus = config.list_physical_devices('GPU')
      if (len(gpus) > 0 and determinism_required and
          x.dtype in (np.float64, np.complex128)):

        with self.assertRaisesRegex(
            errors.UnimplementedError,
            "No deterministic GPU implementation of sparse_dense_matmul "
            "available for data of type tf.float64 or tf.complex128"):
          tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(
              sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
          self.evaluate(tf_value_ans)
      else:
        tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(
            sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        tf_tensor_ans = sparse_ops.sparse_tensor_dense_matmul(
            sparse_tensor.SparseTensor.from_value(sp_x_value),
            y,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b)

        # Ensure that the RHS shape is known at least.
        self.assertEqual(tf_value_ans.get_shape()[1], np_ans.shape[1])
        self.assertEqual(tf_tensor_ans.get_shape()[1], np_ans.shape[1])

        for out in (self.evaluate(tf_value_ans), self.evaluate(tf_tensor_ans)):
          if x.dtype == np.float32:
            self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-4)
          elif x.dtype == np.float64:
            self.assertAllClose(np_ans, out, rtol=1e-6, atol=1e-6)
          elif x.dtype == np.float16:
            self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)
          else:
            self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-4)

  def _testBasic(self, value_dtype, indices_dtype=np.int64):
    x = _maybe_complex(np.random.rand(10, 10).astype(value_dtype))
    x[np.abs(x) < 0.5] = 0  # Make it sparse

    y = _maybe_complex(np.random.randn(10, 20).astype(value_dtype))

    self._testMatmul(x, y, indices_dtype=indices_dtype)

  def testBasic(self):
    np.random.seed(127)  # Repeatable results
    self._testBasic(np.int32)
    self._testBasic(np.float16)
    self._testBasic(np.float32)
    self._testBasic(np.float64)
    self._testBasic(np.complex64)
    self._testBasic(np.complex128)
    self._testBasic(np.int32, indices_dtype=np.int32)
    self._testBasic(np.float32, indices_dtype=np.int32)

  def testShapeInference(self):
    x = np.random.rand(10, 10)
    x[np.abs(x) < 0.5] = 0  # Make it sparse
    y = np.random.randn(10, 20)
    x_indices = np.vstack(np.where(x)).astype(np.int64).T
    x_values = x[np.where(x)]
    x_shape = x.shape

    with ops.Graph().as_default():
      x_st = sparse_tensor.SparseTensor(x_indices, x_values, x_shape)
      result = sparse_ops.sparse_tensor_dense_matmul(x_st, y)
      self.assertEqual(result.get_shape(), (10, 20))

      x_shape_unknown = array_ops.placeholder(dtype=dtypes.int64, shape=None)
      x_st_shape_unknown = sparse_tensor.SparseTensor(x_indices, x_values,
                                                      x_shape_unknown)
      result_left_shape_unknown = sparse_ops.sparse_tensor_dense_matmul(
          x_st_shape_unknown, y)
      self.assertEqual(result_left_shape_unknown.get_shape().as_list(),
                       [None, 20])

      x_shape_inconsistent = [10, 15]
      x_st_shape_inconsistent = sparse_tensor.SparseTensor(
          x_indices, x_values, x_shape_inconsistent)
      with self.assertRaisesRegex(ValueError, "Dimensions must be equal"):
        sparse_ops.sparse_tensor_dense_matmul(x_st_shape_inconsistent, y)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testInvalidIndicesForSparseTensorDenseMatmul(self):
    # TODO(b/169813429): Make GPU kernel return nice errors too.
    indices = np.matrix([[1, 10]]).astype(np.int64)
    values = np.array([10]).astype(np.float32)
    shape = [3, 2]
    sparse_t = sparse_tensor.SparseTensor(indices, values, shape)

    # Test multiplying by both a small and large dense matrix, to hit
    # both cases in the kernel.
    dense_t = np.matrix([[1] * 5, [2] * 5], dtype=np.float32)
    with self.assertRaisesOpError("k .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
    dense_t = np.matrix([[1] * 500, [2] * 500], dtype=np.float32)
    with self.assertRaisesOpError("k .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

    # Repeat with adjoint_a, to get a different error.
    dense_t = np.matrix([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
    with self.assertRaisesOpError("m .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(
          sparse_ops.sparse_tensor_dense_matmul(
              sparse_t, dense_t, adjoint_a=True))
    dense_t = np.matrix([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
    with self.assertRaisesOpError("m .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(
          sparse_ops.sparse_tensor_dense_matmul(
              sparse_t, dense_t, adjoint_a=True))

  @test_util.run_gpu_only
  def testInvalidIndicesForSparseTensorDenseMatmulOnGPU(self):
    indices = np.array([[1, 10]]).astype(np.int64)
    values = np.array([10]).astype(np.float32)
    shape = [3, 2]
    sparse_t = sparse_tensor.SparseTensor(indices, values, shape)

    # Test multiplying by both a small and large dense matrix, to hit
    # both cases in the kernel.
    dense_t = np.matrix([[1] * 5, [2] * 5], dtype=np.float32)
    expected_t = np.array([[0] * 5, [np.nan] * 5, [0] * 5], dtype=np.float32)
    self.assertAllClose(
        expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
    dense_t = np.matrix([[1] * 500, [2] * 500], dtype=np.float32)
    expected_t = np.array([[0] * 500, [np.nan] * 500, [0] * 500],
                          dtype=np.float32)
    self.assertAllClose(
        expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

    # Repeat with adjoint_a, now the error is that the sparse index
    # is OOO w.r.t. the output.  The GPU kernel can't do much here,
    # so it just doesn't accumulate.

    dense_t = np.matrix([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
    expected_t = np.array([[0] * 5, [0] * 5], dtype=np.float32)
    self.assertAllClose(
        expected_t,
        sparse_ops.sparse_tensor_dense_matmul(
            sparse_t, dense_t, adjoint_a=True))

    dense_t = np.matrix([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
    expected_t = np.array([[0] * 500, [0] * 500], dtype=np.float32)
    self.assertAllClose(
        expected_t,
        sparse_ops.sparse_tensor_dense_matmul(
            sparse_t, dense_t, adjoint_a=True))

  # Tests setting one dimension to be a high value.
  def _testLarge(self, np_dtype):
    r1 = np.random.randint(6000, 20000)
    r2 = np.random.randint(1, 10)
    r3 = np.random.randint(1, 10)

    for m, k, n in [(r1, r2, r3),
                    (r2, r1, r3),
                    (r2, r3, r1)]:
      x = _maybe_complex(np.random.rand(m, k).astype(np_dtype))
      x[np.abs(x) < 0.8] = 0

      y = _maybe_complex(np.random.randn(k, n).astype(np_dtype))

      self._testMatmul(x, y, adjoint_a=False, adjoint_b=False)
      self._testMatmul(x.transpose(), y, adjoint_a=True, adjoint_b=False)
      self._testMatmul(x, y.transpose(), adjoint_a=False, adjoint_b=True)
      self._testMatmul(
          x.transpose(), y.transpose(), adjoint_a=True, adjoint_b=True)

    np.random.seed(127)  # Repeatable results
    self._testLarge(np.float32)
    self._testLarge(np.float64)
    self._testLarge(np.complex64)
    self._testLarge(np.complex128)

  # Tests random sized matrices.
  def testFloatRandom(self):
    np.random.seed(127)  # Repeatable results
    for _ in range(8):
      for adjoint_a in [True, False]:
        for adjoint_b in [True, False]:
          for thresh in [0.0, 0.2, 0.8, 1.0]:
            n, k, m = np.random.randint(1, 100, size=3)
            x = np.random.rand(n, k).astype(np.float32)
            x[x < thresh] = 0  # Make it sparse
            y = np.random.randn(k, m).astype(np.float32)
            x = x.transpose() if adjoint_a else x
            y = y.transpose() if adjoint_b else y
            self._testMatmul(x, y, adjoint_a, adjoint_b)

