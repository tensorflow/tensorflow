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

import itertools

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad  # pylint: disable=unused-import
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def dense_to_csr_sparse_matrix(dense):
  dense_t = ops.convert_to_tensor(dense)
  locs = array_ops.stop_gradient(array_ops.where(math_ops.abs(dense_t) > 0))
  return sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(dense_t, locs)


def _add_test(test, op_name, testcase_name, fn):  # pylint: disable=redefined-outer-name
  if fn is None:
    return
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class CSRSparseMatrixDenseMatMulGradTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CSRSparseMatrixDenseMatMulGradTest, cls).setUpClass()
    cls._gpu_available = test_util.is_gpu_available()

  # TODO(penporn): Make these tests runnable on eager mode.
  # (tf.gradients and gradient_checker only run in graph mode.)
  @test_util.run_deprecated_v1
  def _testLargeBatchSparseMatrixMatMulGrad(self, datatype, transpose_a,
                                            transpose_b, adjoint_a, adjoint_b,
                                            transpose_output, conjugate_output):
    if not self._gpu_available:
      return

    sparsify = lambda m: m * (m > 0)
    a_mats_val = sparsify(
        np.random.randn(3, 5, 11) +
        1.j * np.random.randn(3, 5, 11)).astype(datatype)
    if transpose_a or adjoint_a:
      a_mats_val = np.transpose(a_mats_val, (0, 2, 1))
    if adjoint_a:
      a_mats_val = np.conj(a_mats_val)
    b_mats_val = (np.random.randn(3, 11, 13) +
                  1.j * np.random.randn(3, 11, 13)).astype(datatype)
    if transpose_b or adjoint_b:
      b_mats_val = np.transpose(b_mats_val, (0, 2, 1))
    if adjoint_b:
      b_mats_val = np.conj(b_mats_val)
    with self.test_session(use_gpu=True):
      a_mats = ops.convert_to_tensor(a_mats_val, dtype=datatype)
      b_mats = ops.convert_to_tensor(b_mats_val, dtype=datatype)
      a_sm = dense_to_csr_sparse_matrix(a_mats)
      c_mats = sparse_csr_matrix_ops.sparse_matrix_mat_mul(
          a_sm,
          b_mats,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b,
          transpose_output=transpose_output,
          conjugate_output=conjugate_output)
      for [ten, val, nn] in [[a_mats, a_mats_val, "a"],
                             [b_mats, b_mats_val, "b"]]:
        tf_logging.info("Testing gradients for %s" % nn)
        theoretical, numerical = gradient_checker.compute_gradient(
            ten,
            ten.get_shape().as_list(),
            c_mats,
            c_mats.get_shape().as_list(),
            x_init_value=val,
            delta=1e-3)
        self.assertAllClose(theoretical, numerical, atol=1e-3, rtol=1e-3)


# These tests are refactored from sparse_csr_matrix_grad_test to keep its size
# "medium".
for dtype in (np.float32, np.complex64):
  for (t_a, t_b, adj_a, adj_b, t_out,
       conj_out) in itertools.product(*(([False, True],) * 6)):

    def create_mat_mul_test_fn(dtype_, t_a_, t_b_, adj_a_, adj_b_, t_out_,
                               conj_out_):
      # Skip invalid cases.
      if (t_a_ and adj_a_) or (t_b_ and adj_b_):
        return
      # Skip cases where we conjugate real matrices.
      if dtype_ == np.float32 and (adj_a_ or adj_b_ or conj_out_):
        return

      def test_fn(self):
        self._testLargeBatchSparseMatrixMatMulGrad(dtype_, t_a_, t_b_, adj_a_,
                                                   adj_b_, t_out_, conj_out_)

      return test_fn

    name = (
        "_testLargeBatchSparseMatrixMatMulGrad_dtype_%s_t_a_%s_t_b_%s_adj_a_%s_"
        "adj_b_%s_t_out_%s_conj_out_%s" %
        (dtype.__name__, t_a, t_b, adj_a, adj_b, t_out, conj_out))

    _add_test(
        CSRSparseMatrixDenseMatMulGradTest, "CSRSparseMatrixGradTest", name,
        create_mat_mul_test_fn(dtype, t_a, t_b, adj_a, adj_b, t_out, conj_out))

if __name__ == "__main__":
  test.main()
