# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""CSR sparse matrix tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad  # pylint: disable=unused-import
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def dense_to_csr_sparse_matrix(dense):
  dense_t = ops.convert_to_tensor(dense)
  locs = array_ops.where(math_ops.abs(dense_t) > 0)
  return sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(dense_t, locs)


def _add_test(test, op_name, testcase_name, fn):  # pylint: disable=redefined-outer-name
  if fn is None:
    return
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class CSRSparseMatrixGradTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CSRSparseMatrixGradTest, cls).setUpClass()
    cls._gpu_available = test_util.is_gpu_available()

  # TODO(penporn): Make these tests runnable on eager mode.
  # (tf.gradients and gradient_checker only run in graph mode.)
  @test_util.run_deprecated_v1
  def testLargeBatchConversionGrad(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    for dense_shape in ([53, 65, 127], [127, 65]):
      mats_val = sparsify(np.random.randn(*dense_shape))
      with self.test_session() as sess:
        mats = math_ops.cast(mats_val, dtype=dtypes.float32)
        sparse_mats = dense_to_csr_sparse_matrix(mats)
        dense_mats = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            sparse_mats, dtypes.float32)
        grad_vals = np.random.randn(*dense_shape).astype(np.float32)
        grad_out = gradients_impl.gradients([dense_mats], [mats],
                                            [grad_vals])[0]
        self.assertEqual(grad_out.dtype, dtypes.float32)
        self.assertEqual(grad_out.shape, dense_shape)
        grad_out_value = sess.run(grad_out)
        tf_logging.info("testLargeBatchConversionGrad: Testing shape %s" %
                        dense_shape)
        nonzero_indices = abs(mats_val) > 0.0
        self.assertAllEqual(grad_out_value[nonzero_indices],
                            grad_vals[nonzero_indices])
        self.assertTrue(
            np.all(grad_out_value[np.logical_not(nonzero_indices)] == 0.0))

  @test_util.run_deprecated_v1
  def testLargeBatchSparseConversionGrad(self):
    sparsify = lambda m: m * (m > 0)
    for dense_shape in ([53, 65, 127], [127, 65]):
      mats_val = sparsify(np.random.randn(*dense_shape))

      with self.session(use_gpu=True) as sess:
        indices = array_ops.where_v2(
            math_ops.not_equal(mats_val, array_ops.zeros_like(mats_val)))
        values = math_ops.cast(
            array_ops.gather_nd(mats_val, indices), dtype=dtypes.float32)

        grad_vals = np.random.randn(*sess.run(values).shape).astype(np.float32)
        csr_matrix = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            indices, values, dense_shape)
        new_coo_tensor = (
            sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
                csr_matrix, type=dtypes.float32))
        grad_out = gradients_impl.gradients([new_coo_tensor.values], [values],
                                            [grad_vals])[0]
        self.assertEqual(grad_out.dtype, dtypes.float32)
        grad_out_vals = sess.run(grad_out)
        self.assertAllClose(grad_vals, grad_out_vals)

  @test_util.run_deprecated_v1
  def testLargeBatchSparseMatrixAddGrad(self):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    for dense_shape in ([53, 65, 127], [127, 65]):
      a_mats_val = sparsify(np.random.randn(*dense_shape))
      b_mats_val = sparsify(np.random.randn(*dense_shape))
      alpha = np.float32(0.5)
      beta = np.float32(-1.5)
      grad_vals = np.random.randn(*dense_shape).astype(np.float32)
      expected_a_grad = alpha * grad_vals
      expected_b_grad = beta * grad_vals
      expected_a_grad[abs(a_mats_val) == 0.0] = 0.0
      expected_b_grad[abs(b_mats_val) == 0.0] = 0.0
      with self.test_session() as sess:
        a_mats = math_ops.cast(a_mats_val, dtype=dtypes.float32)
        b_mats = math_ops.cast(b_mats_val, dtype=dtypes.float32)
        a_sm = dense_to_csr_sparse_matrix(a_mats)
        b_sm = dense_to_csr_sparse_matrix(b_mats)
        c_sm = sparse_csr_matrix_ops.sparse_matrix_add(
            a_sm, b_sm, alpha=alpha, beta=beta)
        c_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            c_sm, dtypes.float32)
        a_grad, b_grad = gradients_impl.gradients([c_dense], [a_mats, b_mats],
                                                  [grad_vals])
        self.assertEqual(a_grad.dtype, dtypes.float32)
        self.assertEqual(b_grad.dtype, dtypes.float32)
        self.assertEqual(a_grad.shape, dense_shape)
        self.assertEqual(b_grad.shape, dense_shape)
        a_grad_value, b_grad_value = sess.run((a_grad, b_grad))
        tf_logging.info("testLargeBatchConversionGrad: Testing shape %s" %
                        dense_shape)
        self.assertAllEqual(expected_a_grad, a_grad_value)
        self.assertAllEqual(expected_b_grad, b_grad_value)


if __name__ == "__main__":
  test.main()
